#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform DrawPushConstants {
    vec2 viewportSize;
    float lineWidthPx;
    float featherPx;
    uint mode;
} draw;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inBarycentric;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) noperspective out vec3 fragBarycentric;

const uint DRAW_UI = 2u;

void main() {
    if(draw.mode == DRAW_UI) {
        // UI 顶点使用左上角为原点、Y 向下的 framebuffer 像素坐标。
        vec2 ndc = inPosition.xy / draw.viewportSize * 2.0 - 1.0;
        gl_Position = vec4(ndc, 0.0, 1.0);
    } else {
        gl_Position =
            ubo.proj *
            ubo.view *
            ubo.model *
            vec4(inPosition, 1.0);
    }

    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragBarycentric = inBarycentric;
}
