"""
scene3d.py
----------
メインの 3D の場面：机の上の物・ゴールのカゴ・ロボットを、斜め上から見下ろす透視投影で描く。

  ・視点はほぼ固定（SCENE_EYE → SCENE_TARGET）。ロボットを SCENE_FOLLOW の割合だけゆっくり追う
  ・電極劣化のとき、視点を少し揺らす（shake）
  ・影は、物の下の床に置いた半透明の黒い楕円（光源から計算する本物の影はソフトウェア描画では重いので）
  ・物の種類ごとに形と色を変える（箱 = 立方体、缶 = 円柱、玉 = 球、ボーナス = 金色に光る星形）
  ・project()：3D の点 → 画面の画素。視線で選ぶとき、物の位置を画面へ投影して注視点に一番近い物に吸い付くのに使う

座標の対応は robot_view.py の説明を参照（ワールド座標 (x, y) → 3D (x − W/2, 高さ, H/2 − y)）。
"""

import math
import os

import numpy as np
from OpenGL import GL as gl

import config
from gfx2d import rotate_z, scale, translate
from gl_utils import ShaderProgram, look_at, perspective
from robot_view import Meshes, RobotView, load_obj, rotate_y

LIGHT_DIR = np.array([-0.45, 1.0, 0.55])
BASKET_COLOR = (0.75, 0.45, 0.18)

# 質感（画質 standard / high の新しいシェーダー用）：名前 → (鏡面反射の強さ, 鋭さ, 金属らしさ)
# light（以前のシェーダー）では使わず、draw() の spec だけを使う
MATERIALS = {
    "matte":    (0.15, 10.0, 0.0),    # つやなし（ゴム・床）
    "cardboard": (0.22, 14.0, 0.0),   # 段ボール（箱）
    "wood":     (0.30, 24.0, 0.0),    # 机（ニス塗りの木）
    "painted":  (0.45, 32.0, 0.0),    # 塗装（カゴ）
    "plastic":  (0.60, 48.0, 0.0),    # プラスチック（箱）
    "glossy":   (0.90, 120.0, 0.0),   # つやつや（玉）
    "anodized": (0.80, 70.0, 0.55),   # アルマイト（缶・関節）：色のついた金属
    "gunmetal": (0.80, 50.0, 0.80),   # 暗い金属（ロボットの台座）
    "steel":    (1.00, 80.0, 1.00),   # 鋼（アーム・指）
    "gold":     (1.00, 90.0, 1.00),   # 金（ボーナス）
}

ITEM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models", "items.obj")

# 物とカゴのモデル（assets/models/items.obj。tools/make_item_models.py で生成）の役割ごとの見た目：
#   役割 → (色（None = 物の種類の色）, 質感（None = 物の種類の finish）, 発光)
ITEM_PARTS = {
    "main":   (None, None, 0.0),
    "tape":   ((0.50, 0.32, 0.16), "glossy", 0.0),     # 箱：こげ茶の梱包テープ（明るい段ボールの上で見えるように）
    "label":  ((0.93, 0.93, 0.90), "painted", 0.0),    # 箱：白いラベル
    "print":  ((0.10, 0.12, 0.16), "matte", 0.0),      # 箱：印刷の印、缶：飲み口の線
    "band":   ((0.95, 0.96, 0.97), "painted", 0.0),    # 缶：白いラベルの帯
    "metal":  ((0.78, 0.80, 0.84), "steel", 0.0),      # 缶：ふち・ふた
    "tab":    ((0.82, 0.84, 0.88), "steel", 0.0),      # 缶：プルタブ
    "seam":   ((0.97, 0.96, 0.92), "matte", 0.0),      # 玉：縫い目
    "core":   ((1.00, 0.95, 0.70), "gold", 1.4),       # ボーナス：光る芯
    "ring":   ((1.00, 0.85, 0.40), "gold", 0.9),       # ボーナス：まわりを回る輪
    "frame":  (None, "gunmetal", 0.0),                 # カゴ：枠（色は BASKET_COLOR_HQ）
    "mesh":   ((0.50, 0.56, 0.64), "steel", 0.0),      # カゴ：網目
    "bottom": (None, "painted", 0.0),                  # カゴ：底（色は BASKET_COLOR_HQ の暗い色）
    "rim":    (None, "plastic", 0.6),                  # カゴ：上のふちの光る線（色は BASKET_RIM_HQ）
}


def rotate_x(a):
    m = np.identity(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


class Scene3D:
    def __init__(self, rng):
        self.rng = rng
        self.full = config.SCENE_SHADING == "full"
        self.shader = ShaderProgram("scene.vert", "scene_hq.frag" if self.full else "scene.frag")
        self.meshes = Meshes()
        self.robot_view = RobotView(self)
        # 画質 standard / high では作り込んだ物・カゴのモデル、light では以前の簡単な形（重さを変えないため）
        self.item_models = load_obj(ITEM_MODEL_PATH) if self.full else None
        self.follow = np.zeros(3)
        self.shake_t = 0.0
        self.shake_amp = 0.0
        self.view = self.proj = None
        self.eye = np.array(config.SCENE_EYE)

    # --------------------------------------------------------
    @staticmethod
    def to3d(p, h=0.0):
        return np.array([p[0] - config.WORLD_W / 2, h, config.WORLD_H / 2 - p[1]], dtype=np.float64)

    def shake(self, amount=1.0):
        """電極劣化の演出：視点を揺らす"""
        self.shake_t = config.DEGRADE_SHAKE_SEC
        self.shake_amp = 0.03 * amount

    def update_camera(self, robot, dt, aspect):
        goal = self.to3d(robot.pos) * config.SCENE_FOLLOW
        goal[1] = 0.0
        self.follow += (goal - self.follow) * min(1.0, dt * 1.5)
        jitter = np.zeros(3)
        if self.shake_t > 0:
            self.shake_t = max(0.0, self.shake_t - dt)
            k = self.shake_t / config.DEGRADE_SHAKE_SEC
            jitter = self.rng.normal(0, self.shake_amp * k, 3)
        self.eye = np.array(config.SCENE_EYE) + self.follow + jitter
        target = np.array(config.SCENE_TARGET) + self.follow + jitter * 0.5
        self.view = look_at(self.eye, target, (0.0, 1.0, 0.0))
        self.proj = perspective(config.SCENE_FOV, aspect, 0.05, 10.0)

    def project(self, p, h, rect):
        """ワールド座標の点（高さ h）→ 画面の画素（rect = 場面を描いた範囲 (x, y, w, h)、左上原点）"""
        if self.view is None:
            return None
        c = self.proj @ self.view @ np.append(self.to3d(p, h), 1.0)
        if c[3] <= 1e-6:
            return None
        x, y, w, hh = rect
        ndc = c[:3] / c[3]
        return np.array([x + (ndc[0] + 1) / 2 * w, y + (1 - ndc[1]) / 2 * hh])

    def screen_radius(self, p, h, r, rect):
        """ワールド座標の半径 r が、画面でおよそ何画素になるか"""
        a = self.project(p, h, rect)
        b = self.project(np.asarray(p) + [r, 0.0], h, rect)
        return float(np.linalg.norm(b - a)) if a is not None and b is not None else 20.0

    # --------------------------------------------------------
    def draw(self, mesh, model, rot, color, alpha=1.0, emissive=0.0, spec=0.15, material="matte", wood=False):
        """
        spec：以前のシェーダー（画質 light）の鏡面反射の強さ。
        material・wood：新しいシェーダー（standard / high）の質感（MATERIALS の名前）と、木目にするか
        """
        s = self.shader
        s.set_mat4("uModel", model)
        s.set_mat4("uRot", rot)
        s.set_vec3("uColor", color)
        s.set_float("uAlpha", alpha)
        s.set_float("uEmissive", emissive)
        if self.full:
            sp, shine, metal = MATERIALS[material]
            s.set_float("uSpec", sp)
            s.set_float("uShine", shine)
            s.set_float("uMetal", metal)
            s.set_int("uMat", 1 if wood else 0)
        else:
            s.set_float("uSpec", spec)
        gl.glBindVertexArray(mesh[0])
        gl.glDrawElements(gl.GL_TRIANGLES, mesh[1], gl.GL_UNSIGNED_INT, None)

    def _shadow(self, p, r, strength=1.0):
        """床に置いた半透明の楕円（ふたつきの円柱を平たくしたもの）"""
        if strength <= 0.01:
            return
        c = self.to3d(p, 0.0006)
        m = translate(*c) @ rotate_z(math.pi / 2) @ scale(0.0005, r, r)
        self.draw(self.meshes.cyl, m, np.identity(4, dtype=np.float32), (0, 0, 0),
                  alpha=config.SHADOW_ALPHA * strength)

    def _model_parts(self, name, M, rot, item_color=None, item_finish="plastic", glow=0.0, skip=()):
        """items.obj の部品を描く（役割ごとの色・質感は ITEM_PARTS。skip の役割は描かない）"""
        for role, mesh in self.item_models[name]:
            if role in skip:
                continue
            color, material, emissive = ITEM_PARTS[role]
            if role == "main":
                color, material, emissive = item_color, item_finish, glow
            elif name == "basket":
                if role == "frame":
                    color = config.BASKET_COLOR_HQ
                elif role == "bottom":
                    color = tuple(v * 0.55 for v in config.BASKET_COLOR_HQ)
                elif role == "rim":
                    color = config.BASKET_RIM_HQ
            self.draw(mesh, M, rot, color, emissive=emissive, material=material)

    def _item_model(self, it, t, selected):
        """作り込んだモデルで物を描く（大きさ・高さは以前の簡単な形と同じ）"""
        spec = it.spec
        r = spec["radius"]
        glow = 0.35 if selected else 0.0
        name = {"cube": "box", "cylinder": "can", "sphere": "ball", "star": "bonus"}[spec["shape"]]
        S = scale(r, r, r)
        if name == "bonus":                     # 回りながら上下にゆれる。輪は傾けて別に回す
            bob = 0.12 * math.sin(t * 2.6 + it.id)
            rot = rotate_y(it.angle + t * 1.6)
            base = translate(*self.to3d(it.pos, it.height + bob * r))
            pulse = 0.5 + 0.5 * math.sin(t * 5)
            self._model_parts("bonus", base @ rot @ S, rot, spec["rgb"], "gold", 0.12 + 0.18 * pulse + glow,
                              skip=("ring",))
            rr = rotate_y(t * 2.4) @ rotate_x(0.5) @ rotate_y(-t * 3.0)
            ring = base @ translate(0, 1.3 * r, 0) @ rr @ translate(0, -1.3 * r, 0) @ S
            for role, mesh in self.item_models["bonus"]:
                if role == "ring":
                    color, material, emissive = ITEM_PARTS["ring"]
                    self.draw(mesh, ring, rr, color, emissive=emissive + 0.4 * pulse, material=material)
            return
        rot = rotate_y(it.angle)
        M = translate(*self.to3d(it.pos, it.height)) @ rot @ S
        self._model_parts(name, M, rot, spec["rgb"], spec.get("finish", "plastic"), glow)

    def _bonus_glow(self, it, t):
        """ボーナスの光（加算合成）：まわりのぼんやりした光と、立ちのぼって消えていく光の粒"""
        r = it.spec["radius"]
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        c = self.to3d(it.pos, it.height + 1.3 * r)
        gold = (1.0, 0.78, 0.30)
        I = np.identity(4, dtype=np.float32)
        for size, a in ((1.4, 0.08), (1.75, 0.06), (2.15, 0.045), (2.6, 0.033), (3.1, 0.022), (3.7, 0.014)):
            # 外ほど薄い光の玉を重ねる（ぼかしの代わり）
            s = r * size * (1.0 + 0.08 * pulse)
            self.draw(self.meshes.sph, translate(*c) @ scale(s, s, s), I, gold, alpha=a * (0.7 + 0.3 * pulse),
                      emissive=2.5, material="matte")
        # 星のまわりから立ちのぼって消えていく光の粒（8 個）
        for k in range(8):
            ph = (t * 0.55 + k / 8) % 1.0
            ang = k * 2.4 + t * 0.8
            p = np.array(c) + np.array([math.cos(ang) * 1.1 * r, (ph * 3.2 - 1.2) * r, math.sin(ang) * 1.1 * r])
            sz = r * 0.13 * (1.0 - 0.5 * ph)
            self.draw(self.meshes.sph, translate(*p) @ scale(sz, sz, sz), I, (1.0, 0.92, 0.6),
                      alpha=0.9 * math.sin(math.pi * ph), emissive=2.5, material="matte")

    def _item(self, it, t, selected=False):
        """物を 1 個描く（種類ごとの形）。つかまれている物は呼び出し側で位置・高さを決める"""
        if self.item_models is not None:
            return self._item_model(it, t, selected)
        spec = it.spec
        r = spec["radius"]
        h = it.height
        rot = rotate_y(it.angle)
        glow = 0.35 if selected else 0.0
        shape = spec["shape"]
        mat = spec.get("finish", "plastic")
        if shape == "cube":
            c = self.to3d(it.pos, h + r * 0.85)
            self.draw(self.meshes.cube, translate(*c) @ rot @ scale(r * 0.85, r * 0.85, r * 0.85), rot, spec["rgb"],
                      emissive=glow, spec=0.2, material=mat)
        elif shape == "cylinder":
            c = self.to3d(it.pos, h)
            m = translate(*c) @ rot @ rotate_z(math.pi / 2) @ scale(r * 2.6, r, r)
            self.draw(self.meshes.cyl, m, rot @ rotate_z(math.pi / 2), spec["rgb"], emissive=glow, spec=0.7,
                      material=mat)
        elif shape == "sphere":
            c = self.to3d(it.pos, h + r)
            self.draw(self.meshes.sph, translate(*c) @ scale(r, r, r), np.identity(4, dtype=np.float32), spec["rgb"],
                      emissive=glow, spec=0.8, material=mat)
        else:   # star：金色に光り、ゆっくり回る（立方体 2 つを 45° ずらして重ねる）
            c = self.to3d(it.pos, h + r * 1.1)
            pulse = 0.35 + 0.25 * math.sin(t * 5)
            for k in range(2):
                rr = rotate_y(it.angle + t * 1.5 + k * math.pi / 4) @ rotate_z(math.pi / 4)
                self.draw(self.meshes.cube, translate(*c) @ rr @ scale(r, r * 0.35, r), rr, spec["rgb"],
                          emissive=(pulse + glow) * (0.6 if self.full else 1.0), spec=0.9, material=mat)

    def _basket(self):
        """ゴールのカゴ：底と 4 枚の壁（上が開いた箱）"""
        g = np.asarray(config.GOAL_POS)
        R = config.GOAL_RADIUS * 0.92
        hgt = 0.045
        th = 0.006
        c = self.to3d(g, 0.004)
        I = np.identity(4, dtype=np.float32)
        if self.item_models is not None:      # 作り込んだカゴ（網目・ふち・光る線）
            g3 = self.to3d(np.asarray(config.GOAL_POS), 0.0)
            self._model_parts("basket", translate(*g3), np.identity(4, dtype=np.float32))
            return
        wall = config.BASKET_COLOR_HQ if self.full else BASKET_COLOR
        bottom = tuple(v * 0.55 for v in config.BASKET_COLOR_HQ) if self.full else (0.35, 0.22, 0.10)
        self.draw(self.meshes.cube, translate(*c) @ scale(R, 0.004, R), I, bottom, material="painted")
        for dx, dz, sx, sz in ((R, 0, th, R), (-R, 0, th, R), (0, R, R, th), (0, -R, R, th)):
            cc = c + np.array([dx, hgt / 2, dz])
            self.draw(self.meshes.cube, translate(*cc) @ scale(sx, hgt / 2, sz), I, wall, spec=0.2,
                      material="gunmetal")
            if self.full:                          # 上の縁の発光ライン（ゴールの目印）
                rc = c + np.array([dx, hgt + 0.001, dz])
                self.draw(self.meshes.cube, translate(*rc) @ scale(max(sx * 0.6, 0.0035) if sx < R else sx, 0.001,
                                                                  max(sz * 0.6, 0.0035) if sz < R else sz), I,
                          config.BASKET_RIM_HQ, emissive=0.6, material="plastic")

    def _table(self):
        I = np.identity(4, dtype=np.float32)
        m = 0.07
        self.draw(self.meshes.cube, translate(0, -0.03, 0) @ scale(config.WORLD_W / 2 + m, 0.03, config.WORLD_H / 2 + m),
                  I, config.TABLE_COLOR_HQ if self.full else config.TABLE_COLOR, spec=0.08, material="wood", wood=True)
        # 机の上の薄い格子（距離感をつかみやすくする。新しいシェーダーでは木目を邪魔しないよう暗い細い線）
        col = tuple(c * 0.70 for c in config.TABLE_COLOR_HQ) if self.full else \
            tuple(min(1.0, c * 1.25) for c in config.TABLE_COLOR)
        gw = 0.0009 if self.full else 0.0015      # 線の太さの半分
        for k in range(1, 8):
            x = -config.WORLD_W / 2 + config.WORLD_W * k / 8
            self.draw(self.meshes.cube, translate(x, 0.0003, 0) @ scale(gw, 0.0003, config.WORLD_H / 2), I, col)
        for k in range(1, 6):
            z = -config.WORLD_H / 2 + config.WORLD_H * k / 6
            self.draw(self.meshes.cube, translate(0, 0.0003, z) @ scale(config.WORLD_W / 2, 0.0003, gw), I, col)
        # 床（机の下、遠くまで）
        self.draw(self.meshes.cube, translate(0, -0.75, 0) @ scale(4.0, 0.01, 4.0), I,
                  config.SCENE_FLOOR if self.full else (0.10, 0.11, 0.15))

    @staticmethod
    def _set_lights(s):
        """新しいシェーダーの光（config の KEY / FILL / RIM_LIGHT、環境光、フォグ）。色は線形にせず sRGB のまま渡し、
        強さだけ掛ける（シェーダーで線形にするのは物の色だけ。光の色は見た目で調整した値）"""
        for name, (d, col, k) in (("Key", config.KEY_LIGHT), ("Fill", config.FILL_LIGHT), ("Rim", config.RIM_LIGHT)):
            s.set_vec3(f"u{name}Dir", d)
            s.set_vec3(f"u{name}Col", tuple(c * k for c in col))
        s.set_vec3("uSkyCol", tuple(c * config.SKY_AMBIENT[1] for c in config.SKY_AMBIENT[0]))
        s.set_vec3("uGroundCol", tuple(c * config.GROUND_AMBIENT[1] for c in config.GROUND_AMBIENT[0]))
        s.set_vec3("uFogCol", config.SCENE_BG)
        s.set_vec2("uFogRange", config.FOG_RANGE)
        s.set_float("uExposure", config.EXPOSURE)

    # --------------------------------------------------------
    def render(self, viewport, robot, items, held, t, dt, selected=None):
        """viewport：glViewport の値（左下原点、画素）"""
        gl.glViewport(*viewport)
        self.update_camera(robot, dt, viewport[2] / max(viewport[3], 1))
        gl.glEnable(gl.GL_DEPTH_TEST)
        s = self.shader
        s.use()
        s.set_mat4("uView", self.view)
        s.set_mat4("uProj", self.proj)
        s.set_vec3("uLightDir", LIGHT_DIR)
        s.set_vec3("uCamPos", self.eye)
        if self.full:
            self._set_lights(s)

        self._table()
        self._basket()

        # ---- 影（半透明。深度は書かない）----
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        self._shadow(robot.pos, 0.075)
        for it in items:
            if not it.held:
                self._shadow(it.pos, it.radius * 1.3, 1.0 - min(1.0, it.height / 0.6))
        if held is not None:
            self._shadow(held.pos, held.radius * 1.2, 0.6)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        # ---- 物とロボット ----
        for it in items:
            if not it.held:
                self._item(it, t, selected == it.id)
        # 持っている物の表面の、つかむ位置からの距離（指をそこで止める）：箱は半分の幅、ほかは半径
        held_half = None if held is None else held.radius * (0.85 if held.spec["shape"] == "cube" else 1.0)
        self.robot_view.draw(robot, held_half)
        if held is not None:                       # つかんでいる物：グリッパーの先、アームの高さの少し下
            saved = held.height
            held.height = RobotView.ARM_HEIGHT - held.radius * 1.2
            self._item(held, t)
            held.height = saved

        # ---- ボーナスの光（加算合成。深度は書かない）----
        if self.item_models is not None:
            bonus = [it for it in items if it.type == "bonus" and not it.held]
            if bonus:
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
                gl.glDepthMask(gl.GL_FALSE)
                for it in bonus:
                    self._bonus_glow(it, t)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
