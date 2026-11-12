#version 330 core
// 大脳皮質の表面（fsaverage5）の頂点シェーダー
layout(location = 0) in vec3 aPos;      // 頂点の位置（OpenGL の座標に変換ずみ）
layout(location = 1) in vec3 aNormal;   // 頂点の法線
layout(location = 2) in float aSulc;    // 溝の深さ（FreeSurfer の sulc。正 = 溝の奥、負 = 回の上）
layout(location = 3) in float aRegion;  // 光らせる領域の番号（0 = なし、1 運動野, 2 手の領域, 3 体性感覚野, 4 前頭眼野, 5 視覚野）

uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vWorld;
out float vSulc;
flat out int vRegion;

void main() {
    vNormal = aNormal;
    vWorld = aPos;
    vSulc = aSulc;
    vRegion = int(aRegion + 0.5);
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
