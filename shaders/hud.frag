#version 330 core
// HUD のフラグメントシェーダー：OpenCV で描いた RGBA 画像をそのまま貼る
in vec2 vUV;
uniform sampler2D uTex;
out vec4 FragColor;

void main() {
    FragColor = texture(uTex, vUV);
}
