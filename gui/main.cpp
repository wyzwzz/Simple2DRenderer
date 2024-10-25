#pragma once
#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
#include <CGUtils/image.hpp>

using namespace wzz::gl;

using namespace wzz::model;

using namespace wzz::image;

constexpr float PI = wzz::math::PI_f;

constexpr float invPI = wzz::math::invPI<float>;

#ifndef NDEBUG
#define set_uniform_var set_uniform_var_unchecked
#endif


void readPointsFromFile()
{

}

void readLinesFromFile()
{

}

struct LineRenderData
{
    std::vector<vec2f> pos_data;
};
struct LineRenderParams
{
    float antialias;
    float linewidth;
    float miter_limit;

};
struct LineRenderContext
{
    LineRenderData render_data;
    LineRenderParams render_params;
};

struct PointRenderData
{
    std::vector<vec2f> pos_data;
    std::vector<vec2f> center_data;
    std::vector<float> radius;
    std::vector<vec3f> color;
};
struct PointRenderParams
{
    float antialias;
};
struct PointRenderContext
{
    PointRenderData render_data;
    PointRenderParams render_params;
};


class Simpler2DRendererView : public gl_app_t
{
public:
    using gl_app_t::gl_app_t;

private:

    void initialize() override
    {
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0,0,0,0));

    }
    
    void frame() override
    {

    }

    void destroy() override
    {

    }
    
private:

    void renderLines()
    {
        
    }

    void renderPoints()
    {
        
    }

private:
    LineRenderContext m_line_render_context;
    PointRenderContext m_point_render_context;

};


void main()
{
    window_desc_t window_desc;
    window_desc.size = {1200, 720};
    window_desc.title = "Simple2DRendererViewer";

    Simpler2DRendererView(window_desc).run();
    
    return;
}