#version 330 core
// 影の画像（白 = 濃い）を、影の色で半透明に塗る
in vec2 vUV;
in float vAlpha;

uniform sampler2D uTex;
uniform vec3 uColor;

out vec4 FragColor;

void main() {
    FragColor = vec4(uColor, texture(uTex, vUV).r * vAlpha);
}
