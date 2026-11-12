#version 330 core
// 大脳皮質に見立てた曲面の頂点シェーダー
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vNormal;
out vec3 vWorld;
void main() {
    vNormal = aNormal;
    vWorld = aPos;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
