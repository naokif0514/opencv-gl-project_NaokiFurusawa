#version 330 core
// 3D の場面のフラグメントシェーダー（拡散反射 + 半球の環境光 + 鏡面反射 + 発光）。影は uAlpha で半透明の黒を塗る
in vec3 vNormal;
in vec3 vWorld;

uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAlpha;
uniform float uEmissive;   // 0〜1：光っている物（ボーナス・選ばれた物）
uniform float uSpec;       // 鏡面反射の強さ（金属は強く、木は弱く）

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(uLightDir);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 Hv = normalize(L + V);
    float diff = max(dot(N, L), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    float spec = pow(max(dot(N, Hv), 0.0), 32.0) * uSpec;
    vec3 color = uColor * (0.22 + 0.25 * hemi + 0.70 * diff) + vec3(spec) + uColor * uEmissive;
    color = vec3(1.0) - exp(-color * 1.4);
    FragColor = vec4(pow(color, vec3(1.0 / 1.3)), uAlpha);
}
