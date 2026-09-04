#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) noperspective in vec3 fragBarycentric;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D modelSampler;
layout(binding = 2) uniform sampler2D fontSampler;

layout(push_constant) uniform DrawPushConstants {
    vec2 viewportSize;
    float lineWidthPx;
    float featherPx;
    uint mode;
} draw;

const uint DRAW_MODEL_FILL = 0u;
const uint DRAW_MODEL_WIRE = 1u;
const uint DRAW_UI = 2u;
const uint DRAW_DEPTH_ONLY = 3u;

void main() {
    if(draw.mode == DRAW_MODEL_FILL) {
        outColor = texture(modelSampler, fragTexCoord);
        return;
    }

    if(draw.mode == DRAW_MODEL_WIRE) {
        // 重心坐标在三角形任意一条边上都有一个分量为 0。
        // fwidth 把重心坐标的变化量换算成近似屏幕像素宽度。
        vec3 derivative = max(
            fwidth(fragBarycentric),
            vec3(1e-6)
        );

        float innerWidth = max(
            draw.lineWidthPx - draw.featherPx,
            0.0
        );

        float outerWidth =
            draw.lineWidthPx + draw.featherPx;

        vec3 transition = smoothstep(
            derivative * innerWidth,
            derivative * outerWidth,
            fragBarycentric
        );

        float interior = min(
            transition.x,
            min(transition.y, transition.z)
        );

        float alpha = 1.0 - interior;
        if(alpha <= 0.001) {
            discard;
        }

        // 线框永远为纯白，不读取模型纹理。
        outColor = vec4(1.0, 1.0, 1.0, alpha);
        return;
    }

    if(draw.mode == DRAW_UI) {
        float coverage = texture(fontSampler, fragTexCoord).r;
        outColor = vec4(
            fragColor.rgb,
            fragColor.a * coverage
        );
        return;
    }

    // Depth-only pipeline 的 colorWriteMask 为 0。
    // 这里不 discard，确保完整三角形写入深度。
    if(draw.mode == DRAW_DEPTH_ONLY) {
        outColor = vec4(0.0);
        return;
    }

    discard;
}
