#version 330 core
// 大脳皮質の表面（fsaverage5）の頂点シェーダー
layout(location = 0) in vec3 aPos;      // pial での頂点の位置（OpenGL の座標に変換ずみ）
layout(location = 1) in vec3 aNormal;   // pial での頂点の法線
layout(location = 2) in float aSulc;    // 溝の深さ（FreeSurfer の sulc。正 = 溝の奥、負 = 回の上）
layout(location = 4) in vec3 aPos2;     // inflated での同じ頂点の位置
layout(location = 5) in vec3 aNormal2;  // inflated での法線
layout(location = 3) in float aRegion;  // 光らせる領域の番号（0 = なし、1 運動野, 2 手の領域, 3 体性感覚野, 4 前頭眼野, 5 視覚野）

uniform mat4 uView;
uniform mat4 uProj;
uniform float uMorph;     // 0 = pial、1 = inflated。途中の値で 2 つの表面をなめらかに補間する（変形アニメーション）

out vec3 vNormal;
out vec3 vWorld;
out float vSulc;
flat out int vRegion;

void main() {
    vec3 pos = mix(aPos, aPos2, uMorph);
    vNormal = normalize(mix(aNormal, aNormal2, uMorph));
    vWorld = pos;
    vSulc = aSulc;
    vRegion = int(aRegion + 0.5);
    gl_Position = uProj * uView * vec4(pos, 1.0);
}
