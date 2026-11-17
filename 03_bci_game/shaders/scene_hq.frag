#version 330 core
// 3D の場面のフラグメントシェーダー（画質 standard / high 用。light は scene.frag）
//
//  ・色は sRGB で受け取り、線形にしてから光を計算する（最後にトーンマップ + ガンマ補正で sRGB に戻す）
//  ・光：主な光（暖色）+ 補助の光（寒色）+ 輪郭の光（奥から）+ 半球の環境光（上 = 空、下 = 机の照り返し）
//  ・鏡面反射：Blinn-Phong（正規化つき）+ Schlick の Fresnel（斜めから見るほど強く映る）
//  ・金属（uMetal）：拡散反射をなくし、反射の色を物の色にする。周りの明るさの映り込み（簡易な環境マップ）を足す
//  ・机（uMat = 1）：板張りの木目を、位置から計算で作る（画像を使わない）
//  ・遠くは背景色に溶け込ませる（フォグ）。影（uAlpha < 1 の黒）はそのまま半透明で塗る
in vec3 vNormal;
in vec3 vWorld;

uniform vec3 uColor;
uniform vec3 uCamPos;
uniform float uAlpha;
uniform float uEmissive;   // 0〜1：光っている物（ボーナス・選ばれた物・ロボットの印）
uniform float uSpec;       // 鏡面反射の強さ
uniform float uShine;      // 鏡面反射の鋭さ（大きいほど小さく鋭いハイライト = つるつる）
uniform float uMetal;      // 0 = 非金属、1 = 金属
uniform int uMat;          // 0 = 普通、1 = 木目（机）

uniform vec3 uKeyDir;  uniform vec3 uKeyCol;
uniform vec3 uFillDir; uniform vec3 uFillCol;
uniform vec3 uRimDir;  uniform vec3 uRimCol;
uniform vec3 uSkyCol;  uniform vec3 uGroundCol;
uniform vec3 uFogCol;
uniform vec2 uFogRange;
uniform float uExposure;

out vec4 FragColor;

vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}

// 板張りの木目：奥行き方向（z）に幅 PLANK の板を並べ、木目は左右（x）に流れる
vec3 wood(vec3 base, vec3 p) {
    const float PLANK = 0.105;
    float k = floor(p.z / PLANK);
    float h = hash(vec2(k, 7.3));                         // 板ごとのばらつき
    float along = p.x + h * 3.1;                          // 板ごとに木目の位置をずらす
    float across = p.z - k * PLANK;
    float warp = vnoise(vec2(along * 9.0, across * 40.0)) * 0.9 + vnoise(vec2(along * 30.0, across * 120.0)) * 0.3;
    float grain = fract(across * 110.0 + warp * 2.0);       // 年輪の筋
    grain = smoothstep(0.0, 0.45, grain) * smoothstep(1.0, 0.55, grain);
    vec3 c = base * (0.84 + 0.24 * h) * mix(0.86, 1.04, grain);
    c *= 0.90 + 0.14 * vnoise(vec2(along * 2.5, across * 12.0 + k));   // 大きなむら
    float edge = min(across, PLANK - across);             // 板の継ぎ目（細い暗い線）
    c *= mix(0.62, 1.0, smoothstep(0.0, 0.0022, edge));
    return c;
}

vec3 light(vec3 L, vec3 col, vec3 N, vec3 V, vec3 diffCol, vec3 F0, float shine) {
    L = normalize(L);
    float NdotL = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
    float norm = (shine + 8.0) / 25.13;                   // 鋭いハイライトほど明るく（エネルギーを保つ近似）
    vec3 spec = F * pow(NdotH, shine) * norm * uSpec;
    return (diffCol * (1.0 - F) + spec) * col * NdotL;
}

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCamPos - vWorld);
    vec3 albedo = toLinear(uColor);
    if (uMat == 1) albedo = wood(albedo, vWorld);

    vec3 F0 = mix(vec3(0.04), albedo, uMetal);
    vec3 diffCol = albedo * (1.0 - uMetal);

    // 半球の環境光
    float hemi = 0.5 + 0.5 * N.y;
    vec3 col = diffCol * mix(uGroundCol, uSkyCol, hemi);

    col += light(uKeyDir, uKeyCol, N, V, diffCol, F0, uShine);
    col += light(uFillDir, uFillCol, N, V, diffCol, F0, uShine);
    col += light(uRimDir, uRimCol, N, V, diffCol, F0, uShine);

    // 周りの明るさの映り込み（金属・つやのある物）：上は明るい空、下は暗い。主な光の方向に明るい窓
    vec3 R = reflect(-V, N);
    vec3 env = mix(uGroundCol * 0.5, uSkyCol * 1.6, smoothstep(-0.3, 0.7, R.y));
    env += uKeyCol * 0.35 * pow(max(dot(R, normalize(uKeyDir)), 0.0), 24.0);
    float NdotV = max(dot(N, V), 0.0);
    vec3 Fenv = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0) * clamp(uShine / 96.0, 0.0, 1.0);
    col += env * Fenv * uSpec;

    col += albedo * uEmissive * 1.6;

    // 遠くは背景色に溶け込む
    float d = length(uCamPos - vWorld);
    col = mix(col, toLinear(uFogCol), smoothstep(uFogRange.x, uFogRange.y, d));

    // トーンマップ（ACES の近似）とガンマ補正
    col *= uExposure;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    FragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), uAlpha);
}
