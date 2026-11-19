#version 330 core
// 柔らかい影（画質 standard）：机の上に置いた四角形に、影の画像を貼る。頂点は毎フレーム CPU で作る（shadows.py）
layout(location = 0) in vec3 aPos;     // 3D の位置（机の少し上）
layout(location = 1) in vec2 aUV;      // 影の画像の座標
layout(location = 2) in float aAlpha;  // この影の濃さ

uniform mat4 uView;
uniform mat4 uProj;

out vec2 vUV;
out float vAlpha;

void main() {
    vUV = aUV;
    vAlpha = aAlpha;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
