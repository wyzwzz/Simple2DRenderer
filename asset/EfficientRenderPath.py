import glm
import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44, matrix44
import time
from copy import deepcopy
import cv2

vertex = """
#version 330
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;
in vec2 position;
out float v_antialias;
out float v_linewidth;
out float v_miter_limit;
void main()
{
    v_antialias = antialias;
    v_linewidth = linewidth;
    v_miter_limit = miter_limit;
    gl_Position = vec4(position, 0.0, 1.0);
} """

fragment = """
#version 330
vec4 stroke(float distance, float linewidth, float antialias, vec4 color)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;
    alpha = exp(-alpha*alpha);
    if( border_distance > (linewidth/2.0 + antialias) )
        discard;
    else if( border_distance < 0.0 )
        frag_color = color;
    else
        frag_color = vec4(color.rgb, color.a * alpha);
    return frag_color;
}
vec4 cap(int type, float dx, float dy, float linewidth, float antialias, vec4 color)
{
    float d = 0.0;
    dx = abs(dx);
    dy = abs(dy);
    float t = linewidth/2.0 - antialias;
    // None
    if      (type == 0)  discard;
    // Round
    else if (type == 1)  d = sqrt(dx*dx+dy*dy);
    // Triangle in
    else if (type == 3)  d = (dx+abs(dy));
    // Triangle out
    else if (type == 2)  d = max(abs(dy),(t+dx-abs(dy)));
    // Square
    else if (type == 4)  d = max(dx,dy);
    // Butt
    else if (type == 5)  d = max(dx+t,dy);
    return stroke(d, linewidth, antialias, color);
}
uniform vec4  color;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;
in float v_length;
in vec2  v_caps;
in vec2  v_texcoord;
in vec2  v_bevel_distance;
out vec4 fragColor;
void main()
{
    float distance = v_texcoord.y;
    if (v_caps.x < 0.0)
    {
        fragColor = cap(1, v_texcoord.x, v_texcoord.y, linewidth, antialias, color);
        return;
    }
    if (v_caps.y > v_length)
    {
        fragColor = cap(1, v_texcoord.x-v_length, v_texcoord.y, linewidth, antialias, color);
        return;
    }
    // Round join (instead of miter)
    if (miter_limit < 0) {
        if (v_texcoord.x < 0.0)
        {
            distance = length(v_texcoord);
        }
        else if(v_texcoord.x > v_length)
        {
            distance = length(v_texcoord - vec2(v_length, 0.0));
        }
    } else {
    // Miter limit
    float t = (miter_limit-1.0)*(linewidth/2.0) + antialias;
    if( (v_texcoord.x < 0.0) && (v_bevel_distance.x > (abs(distance) + t)) )
    {
        distance = v_bevel_distance.x - t;
    }
    else if( (v_texcoord.x > v_length) && (v_bevel_distance.y > (abs(distance) + t)) )
    {
        distance = v_bevel_distance.y - t;
    }
    }
    fragColor = stroke(distance, linewidth, antialias, color);
} """

geometry = """
#version 330
layout(lines_adjacency) in; // 4 points at the time from vertex shader
layout(triangle_strip, max_vertices = 4) out; // Outputs a triangle strip with 4 vertices
uniform mat4 projection;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;
//in float v_antialias[];
//in float v_linewidth[];
//in float v_miter_limit[];
out vec2 v_caps;
out float v_length;
out vec2 v_texcoord;
out vec2 v_bevel_distance;
float compute_u(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    // Then  u *= length(p1-p0)
    vec2 v = p1 - p0;
    float l = length(v);
    return ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l;
}
float line_distance(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    vec2 v = p1 - p0;
    float l2 = v.x*v.x + v.y*v.y;
    float u = ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l2;
    // h is the projection of p on (p0,p1)
    vec2 h = p0 + u*v;
    return length(p-h);
}
void main(void)
{
    //float antialias = v_antialias[0];
    //float linewidth = v_linewidth[0];
    //float miter_limit = v_miter_limit[0];
    // Get the four vertices passed to the shader
    vec2 p0 = gl_in[0].gl_Position.xy; // start of previous segment
    vec2 p1 = gl_in[1].gl_Position.xy; // end of previous segment, start of current segment
    vec2 p2 = gl_in[2].gl_Position.xy; // end of current segment, start of next segment
    vec2 p3 = gl_in[3].gl_Position.xy; // end of next segment
    // Determine the direction of each of the 3 segments (previous, current, next)
    vec2 v0 = normalize(p1 - p0);
    vec2 v1 = normalize(p2 - p1);
    vec2 v2 = normalize(p3 - p2);
    // Determine the normal of each of the 3 segments (previous, current, next)
    vec2 n0 = vec2(-v0.y, v0.x);
    vec2 n1 = vec2(-v1.y, v1.x);
    vec2 n2 = vec2(-v2.y, v2.x);
    // Determine miter lines by averaging the normals of the 2 segments
    vec2 miter_a = normalize(n0 + n1); // miter at start of current segment
    vec2 miter_b = normalize(n1 + n2); // miter at end of current segment
    // Determine the length of the miter by projecting it onto normal
    vec2 p,v;
    float d;
    float w = linewidth/2.0 + antialias;
    v_length = length(p2-p1);
    float length_a = w / dot(miter_a, n1);
    float length_b = w / dot(miter_b, n1);
    float m = miter_limit*linewidth/2.0;
    // Angle between prev and current segment (sign only)
    float d0 = -sign(v0.x*v1.y - v0.y*v1.x);
    // Angle between current and next segment (sign only)
    float d1 = -sign(v1.x*v2.y - v1.y*v2.x);
    // Generate the triangle strip
    // First vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 + w*n1;
        v_texcoord = vec2(-w, +w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1 + length_a * miter_a;
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
        v_caps.x = 1.0;
    }
    if( p2 == p3 ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = +d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();
    // Second vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if( p0 == p1 ) {
        p = p1 - w*v1 - w*n1;
        v_texcoord = vec2(-w, -w);
        v_caps.x = v_texcoord.x;
    // Regular join
    } else {
        p = p1 - length_a * miter_a;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
        v_caps.x = 1.0;
    }
    if( p2 == p3 ) v_caps.y = v_texcoord.x;
    else           v_caps.y = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = -d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();
    // Third vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 + w*n1;
        v_texcoord = vec2(v_length+w, +w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2 + length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), +w);
        v_caps.y = 1.0;
    }
    if( p0 == p1 ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = +d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();
    // Fourth vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if( p2 == p3 ) {
        p = p2 + w*v1 - w*n1;
        v_texcoord = vec2(v_length+w, -w);
        v_caps.y = v_texcoord.x;
    // Regular join
    } else {
        p = p2 - length_b * miter_b;
        v_texcoord = vec2(compute_u(p1,p2,p), -w);
        v_caps.y = 1.0;
    }
    if( p0 == p1 ) v_caps.x = v_texcoord.x;
    else           v_caps.x = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = -d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();
    EndPrimitive();
}
"""

class LineRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.BLEND)

        self.prog = self.ctx.program(
            vertex_shader=vertex,
            fragment_shader=fragment,
            geometry_shader=geometry
        )

        self.prog['projection'].write(Matrix44.orthogonal_projection(
            0, self.width, 0, self.height, 0.5, -0.5, dtype='f4'
        ))



    def __getstate__(self):
        """This method is called when pickling."""
        state = self.__dict__.copy()
        del state['ctx']
        del state['prog']
        del state['fbo']
        return state

    def __setstate__(self, state):
        """This method is called when unpickling."""
        self.__dict__.update(state)  # Restore the object's state
        self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.BLEND)
        self.prog = self.ctx.program(
            vertex_shader=vertex,
            fragment_shader=fragment,
            geometry_shader=geometry
        )
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.prog['projection'].write(Matrix44.orthogonal_projection(
            0, self.width, 0, self.height, 0.5, -0.5, dtype='f4'
        ))


    def render_lines(self, all_lines):

        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))

        self.fbo.use()
        self.fbo.clear(0, 0, 0, 1.0)


        all_lines_radius = np.array([float(ll[1])  for ll in all_lines])
        all_lines_colors = [ll[2]  for ll in all_lines]

        all_radius = set(all_lines_radius)
        all_colors = set(all_lines_colors)

        for rr in all_radius:
            for cc in all_colors:
                select_line_idx = np.where((all_lines_radius == rr) & (np.all(np.array(all_lines_colors) == np.array(cc), axis=1) ))[0]
                lines = [all_lines[ii] for ii in select_line_idx]
                lines_vs = [pp for ll in lines for pp in ll[0]]
                lines_strings = np.cumsum([0] + [len(ll[0]) for ll in lines])
                lines_idxs = [j for t in range(len(lines_strings) - 1) for i in
                              range(lines_strings[t], lines_strings[t + 1] - 1) for j in [i, i, i + 1, i + 1]]
                color = lines[0][2]
                P = np.array(lines_vs, dtype=np.float32)
                I = np.array(lines_idxs, dtype= np.int32)
                vbo = self.ctx.buffer(P.astype('f4'))
                ibo = self.ctx.buffer(I.astype('i4'))
                vao = self.ctx.simple_vertex_array(self.prog, vbo, 'position', index_buffer=ibo)

                self.prog['linewidth'].value = rr
                self.prog['antialias'].value = 0
                self.prog['miter_limit'].value = -1
                self.prog['color'].value = (*color, 1)  # Assuming color is RGB, adding alpha=1
                vao.render(moderngl.LINES_ADJACENCY)
                vao.release()
                vbo.release()
                ibo.release()
        self.image = np.array(Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1))
        self.fbo.release()
        # self.ctx.release()


    def __del__(self):
        self.ctx.release()


    def get_image(self):
        return  self.image

    def save_image(self, filename):
        self.get_image().save(filename)

import torch
from EfficientRenderKeepout import KeepoutImageRenderer
from EfficientRenderPoint import KeepoutPointImageRenderer
def print_gpu_memory():
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        print(f"Current GPU memory usage: {current_memory:.2f} MB")
        print(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    else:
        print("CUDA is not available. Are you using a GPU?")


class MultiLevelImageRender:
    def __init__(self, keepout_points, path_starts_ends, path_starts_ends_dirs, board_size, path_width, endpoint_size, resolution = 1024):
        self.keepout_vertices = keepout_points[:, :2]
        self.keepout_vertices_radius = keepout_points[: , 2]
        self.board_size = board_size
        self.path_size = path_width
        self.endpoint_size = endpoint_size
        self.resolution = resolution
        self.path_starts_ends = path_starts_ends
        self.path_starts_ends_dirs = path_starts_ends_dirs * 10
        self.path_starts_ends_paths = [[self.path_starts_ends[i], self.path_starts_ends[i]+self.path_starts_ends_dirs[i]] for i in range(len(self.path_starts_ends))]


        radius = [keepout_points[:, 2][np.where(np.all(keepout_points[:, :2] == pp, axis=1))[0]] for pp in
                           self.path_starts_ends]
        for i in range(len(radius)):
            if len(radius[i]) == 0:
                radius[i] = np.array([1.3])
        radius = np.array(radius)

        self.path_starts_ends = np.concatenate([self.path_starts_ends, radius], axis=1)


        self.image_render = KeepoutPointImageRenderer(size=(int(board_size[0]), int(board_size[1])))
        used_keepout_points = keepout_points.tolist() + self.path_starts_ends.tolist()
        used_keepout_points_colors = [(1, 0, 0)] * len(keepout_points) + [(0, 0, 1)] * len(self.path_starts_ends)
        self.image_render.render_circles(used_keepout_points, used_keepout_points_colors)


        # self.image_render = KeepoutImageRenderer(size=board_size,)
        # self.image_render.set_data(keepout_points, self.path_starts_ends, self.path_starts_ends_paths)
        # self.image_render.update_visible_elements()

        self.routing_start_path_render = LineRenderer(int(board_size[0]), int(board_size[1]))
        dir_lines = [([tuple(( i).tolist()) for i in np.array(pp).astype(np.int32)], radius[pp_i], (0, 0, 1)) for pp_i, pp in enumerate(self.path_starts_ends_paths)]
        self.routing_start_path_render.render_lines(dir_lines)
        routing_start_line_image = self.routing_start_path_render.get_image()
        temp = self.image_render.board_img.copy()
        temp[np.where(routing_start_line_image)] = 255
        self.image_render.board_img = temp

        self.path_render = LineRenderer(self.resolution, self.resolution)

    def render_multi_level_image(self, current_path, previous_paths, routing_pair):
        # print("render_multi_level_image")
        try:
            resolution_size = self.resolution

            start = (current_path[-1] - resolution_size // 2).astype(np.int32)
            end = (current_path[-1] + resolution_size // 2).astype(np.int32)
            keepout_background = self.image_render.render_image([-end[1], -start[1]], [start[0], end[0]])
            routing_pair = routing_pair[:, ::-1].copy()
            out_image = deepcopy(keepout_background[:, :, :3])
            if len(current_path) > 1:
                lines = [([tuple((resolution_size-i).tolist()) for i in (current_path - start).astype(np.int32)], self.path_size, (0, 1, 0))] +\
                        [([tuple((resolution_size-i).tolist()) for i in (pp - start).astype(np.int32)], self.path_size, (1, 0, 0)) for pp in previous_paths]
                self.path_render.render_lines(lines)
                line_image = self.path_render.get_image()
                out_image = keepout_background[:, :, :3] + line_image[::-1, ::-1]
            elif len(previous_paths) > 0:
                lines = [([tuple((resolution_size-i).tolist()) for i in (pp - start).astype(np.int32)], self.path_size, (1, 0, 0)) for pp in previous_paths]
                self.path_render.render_lines(lines)
                line_image = self.path_render.get_image()
                out_image = keepout_background[:, :, :3] + line_image[::-1,::-1]
            out_image = cv2.circle(out_image, (resolution_size // 2, resolution_size // 2), 3, (0, 255, 0), -1)
        except:
            print("Asdf")
        return out_image






if __name__ == '__main__':
    renderer = LineRenderer(800, 800)
    lines = [
        ([(-100, 100), (700, 0),  (700, 700)], 3, (0, 1, 0)),  # Red diagonal line
        ([(100, 700), (700, 100)], 3, (0, 1, 0)),
        ([(50, 700), (50, 100)], 3, (0, 1, 0)),# Green diagonal line
    ]
    start_time = time.time()
    renderer.render_lines(lines)
    end_time = time.time()


    print(end_time - start_time)

    t = renderer.get_image()
    from utils.util import *
    show_float_image(t.astype(np.float32) / 255)
    renderer.save_image('rendered_lines.png')
