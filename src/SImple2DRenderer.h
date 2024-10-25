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

// camera, viewport, resolution, etc...
struct RenderParams
{

};

class Simple2DRenderer
{
public:
    Simple2DRenderer();

    virtual ~Simple2DRenderer();

    virtual bool beginFrame() = 0;

    virtual void endFrame() = 0;

    virtual void readbackFrame() = 0;

    virtual void renderPoints(PointRenderContext context) = 0;

    virtual void renderLines(LineRenderContext context) = 0;

    virtual void setRenderParams(RenderParams render_params) = 0;


};