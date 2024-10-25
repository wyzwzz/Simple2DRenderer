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