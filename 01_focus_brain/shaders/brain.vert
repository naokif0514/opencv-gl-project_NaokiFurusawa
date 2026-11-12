#version 330 core
// 脳オブジェクトの頂点シェーダー
// 頂点の位置と法線をワールド座標に変換して、フラグメントシェーダーに渡す。

layout(location = 0) in vec3 aPos;      // 頂点の位置（モデル座標）
layout(location = 1) in vec3 aNormal;   // 頂点の法線（モデル座標）
layout(location = 2) in float aGroove;  // 溝の深さ 0〜1（溝ほど 1）

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vObjPos;     // 回転前の位置（表面の模様を物体に貼り付けるために使う）
out float vGroove;

void main() {
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos = world.xyz;
    // 回転と一様な拡大だけなので、モデル行列の左上 3x3 で法線を変換してよい
    vNormal = mat3(uModel) * aNormal;
    vObjPos = aPos;
    vGroove = aGroove;
    gl_Position = uProj * uView * world;
}
