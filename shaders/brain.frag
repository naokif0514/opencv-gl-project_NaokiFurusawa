#version 330 core
// 脳オブジェクトのフラグメントシェーダー
// 集中度 uFocus (0〜1) に応じて見た目を変える。
//   ・集中度が低い : 色がくすみ（彩度が下がり）、発光も弱い
//   ・集中度が高い : 鮮やかな色になり、溝と輪郭（リム）が強く発光する
// 陰影は Blinn-Phong（拡散反射 + 鏡面反射）、輪郭の光はフレネル項で表現している。

in vec3 vWorldPos;
in vec3 vNormal;
in vec3 vObjPos;
in float vGroove;

uniform float uFocus;      // 集中度 0〜1（描画側でなめらかにした値）
uniform float uTime;       // 経過時間 [秒]
uniform vec3 uCamPos;      // カメラ位置（ワールド座標）
uniform vec3 uLightDir;    // 光源の方向（光が向かう向きの逆＝光源へ向かう向き）

out vec4 FragColor;

// 彩度を下げる（くすませる）：輝度との線形補間
vec3 desaturate(vec3 c, float amount) {
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    return mix(c, vec3(l), amount);
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(L + V);
    float f = clamp(uFocus, 0.0, 1.0);

    // ---- 表面の基本色 ----
    // 脳らしいピンク寄りの色。溝の部分は少し暗くする
    vec3 baseColor = vec3(0.78, 0.55, 0.62);
    baseColor *= mix(1.0, 0.55, vGroove);
    // 集中度が低いほど彩度を下げて暗くする
    baseColor = desaturate(baseColor, 0.85 * (1.0 - f)) * mix(0.55, 1.0, f);

    // ---- 照明（Blinn-Phong）----
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 48.0) * mix(0.10, 0.35, f);
    vec3 ambient = vec3(0.10, 0.10, 0.14);
    vec3 color = baseColor * (ambient + 0.85 * diff) + vec3(spec);

    // ---- 発光（集中度に応じて強くなる）----
    // 発光色：低いときは青、高いときはシアン〜白っぽく
    vec3 glowColor = mix(vec3(0.20, 0.35, 1.00), vec3(0.35, 1.00, 0.95), f);
    // ゆっくり脈打つ（呼吸のような明滅）
    float breathe = 0.85 + 0.15 * sin(uTime * 2.2);
    float glowStrength = pow(f, 1.6) * breathe;

    // 溝の中から光が漏れるような発光
    vec3 emission = glowColor * vGroove * glowStrength * 1.2;
    // 輪郭（視線と面が平行に近いところ）ほど光るフレネル項
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    emission += glowColor * fresnel * (0.15 + 1.6 * glowStrength);

    color += emission;

    // ---- トーンマッピングとガンマ補正 ----
    color = vec3(1.0) - exp(-color * 1.4);
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, 1.0);
}
