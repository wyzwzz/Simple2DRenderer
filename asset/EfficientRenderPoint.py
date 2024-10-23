import moderngl
import numpy as np
from PIL import Image
import numpy as np
from vispy import app, scene, io
from vispy.scene import visuals
import torch


def create_gradient_heatmap(image_size, start_point, end_point):
    x, y = torch.meshgrid(torch.arange(image_size[0], 0, -1, device="cuda"), torch.arange(image_size[1], device="cuda"), indexing='ij')
    grid = torch.stack((x, y), dim=-1).float()
    end_point = torch.tensor(end_point, device="cuda").float()
    dist_end = torch.sqrt(torch.sum((grid - end_point) ** 2, dim=-1))
    heatmap  = (dist_end / dist_end.max() - 0.5) * 2
    return heatmap.cpu().numpy()


class KeepoutPointImageRenderer:
    def __init__(self, size):
        self.width, self.height = size
        self.size = size
        self.ctx = moderngl.create_standalone_context()
        self.vertex_shader = '''
        #version 330
        
        in vec2 in_position;
        in vec2 in_center;
        in float in_radius;
        in vec3 in_color;
        
        uniform vec2 u_resolution;
        
        out vec2 v_uv;
        out vec2 v_center;
        out float v_radius;
        out vec3 v_color;
        
        void main() {
            v_uv = in_position;
            v_center = in_center;
            v_radius = in_radius;
            v_color = in_color;
        
            // Transform to NDC
            vec2 ndc_position = (in_position * in_radius + in_center) / u_resolution * 2.0 - 1.0;
            gl_Position = vec4(ndc_position, 0.0, 1.0);
        }
        '''
        self.fragment_shader = '''
        #version 330

        in vec2 v_uv;
        in vec2 v_center;
        in float v_radius;
        in vec3 v_color;
        out vec4 f_color;
        void main() {
            float dist = length(v_uv);
            // Improved antialiasing
            float antialias = 0.1; // Adjust this value to control the antialiasing width
            float edge0 = 1.0 - antialias;
            float edge1 = 1.0 + antialias;
        
            float alpha = smoothstep(edge1, edge0, dist);
            
            // Discard fragments outside the circle (including antialiased edge)
            if (alpha == 0.0) {
                discard;
            }
            f_color = vec4(v_color, alpha);
        }
        '''
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)


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
        self.prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader)
        self.fbo = self.ctx.simple_framebuffer(self.size)




    def render_circles(self, circle_data, circle_data_color):

        self.circle_vertices = np.array([
            (-1, -1), (1, -1), (1, 1),
            (-1, -1), (1, 1), (-1, 1)
        ], dtype='f4')

        self.vbo = self.ctx.buffer(self.circle_vertices)
        self.fbo = self.ctx.simple_framebuffer(self.size)

        self.fbo.use()
        self.fbo.clear(0, 0, 0, 1.0)
        self.prog['u_resolution'].value = (self.width, self.height)

        all_points_centers = np.array(circle_data)[:, :2].astype('f4')
        all_points_radius = np.array(circle_data)[:, 2].astype('f4') / 2
        all_points_colors = np.array(circle_data_color).astype('f4')

        center_vbo = self.ctx.buffer(all_points_centers)
        radius_vbo = self.ctx.buffer(all_points_radius)
        color_vbo = self.ctx.buffer(all_points_colors)

        # Create VAO
        vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo, '2f', 'in_position'),
                (center_vbo, '2f/i', 'in_center'),
                (radius_vbo, 'f/i', 'in_radius'),
                (color_vbo, '3f/i', 'in_color')
            ]
        )

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(instances=len(circle_data))

        # Read pixels
        pixels = self.fbo.read(components=4, alignment=1)
        self.board_img  = Image.frombytes('RGBA', self.fbo.size, pixels).transpose(Image.FLIP_TOP_BOTTOM)
        self.board_img  = np.array(self.board_img)

        vao.release()
        self.vbo.release()
        center_vbo.release()
        radius_vbo.release()
        color_vbo.release()
        self.fbo.release()
        self.ctx.release()

    def render_image(self, x_range, y_range, filename=None):
        return self.board_img[x_range[0]:x_range[1], y_range[0]:y_range[1]]





if __name__ == "__main__":
    print("asdf ")