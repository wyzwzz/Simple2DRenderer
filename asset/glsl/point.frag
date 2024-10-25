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