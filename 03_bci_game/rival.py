"""
rival.py
--------
チャレンジモードの CPU のライバルロボット。

■ ライバルも擬似 BCI で動く
  ライバルの「頭（AI）」は、「手先をどちらへ動かしたいか（手の速度）」と「握りたいか（0〜1）」だけを決める。
  それがライバル専用の擬似ニューロン（cortex_sim.CortexSim。数とノイズは強さで変わる）でスパイクになり、
  ライバル専用のデコーダで読み取った速度でアームを動かし、読み取った把持の値で（プレイヤーと同じヒステリシス・
  0.3 秒続いたら放す判定で）グリッパーを開閉する。だから、ニューロンが少ない・ノイズが多いライバルは、
  手先がふらつき、握るタイミングがずれて、たまにつかみ損ねる。
  デコーダは、ゲームの開始時に、プレイヤーと同じ同期フェーズ（hand_calib.py の課題）を画面に出さずに
  一瞬で模擬して学習する（ライバルの「手」は課題のターゲットどおりに動いたとする）。

■ 行動（状態）
  think（次に狙う物を決める）→ go（台座を物の手前へ）→ reach（アームを物へ）→ grip（握る）
  → carry（自分のカゴへ）→ place（カゴの上で放す）→ think …
  ・狙う物：点数 × 自分のカゴからの距離の倍率 ÷（今の場所から物まで + 物からカゴまでの距離）が大きい物。
    プレイヤーのロボットのすぐそばの物は避ける
  ・横取り：プレイヤーがロックオンした物を、強さに応じた確率（steal）で狙いにいく
  ・つかまれている物は取れない。カゴから物を盗むことはしない
  ・プレイヤーのロボットが前にいたら、横へよけて進む（ロボットどうしは separate() で重ならないように押し合う）
■ 見つめて止める（視線で反撃）：freeze() で FREEZE_SEC 秒止まる（物を持っていたら、持ったまま止まる）。
  止まっている間はニューロンも動かさない。止まり終わってから FREEZE_COOLDOWN 秒は止められない
"""

import math
from types import SimpleNamespace

import numpy as np

import config
from cortex_sim import CortexSim
from hand_calib import HandCalibration, grip_target, target_velocity
from mission import GripSwitch, gripper_tip, item_points
from robot import Robot


def separate(a, b, b_fixed=False):
    """ロボット a と b が重なっていたら押し離す（b_fixed なら a だけを動かす：止まっているライバル）"""
    d = a.pos - b.pos
    n = float(np.linalg.norm(d))
    gap = 2 * config.ROBOT_RADIUS - n
    if gap <= 0:
        return False
    u = d / n if n > 1e-6 else np.array([1.0, 0.0])
    m = 0.03
    if b_fixed:
        a.pos += u * gap
    else:
        a.pos += u * gap / 2
        b.pos -= u * gap / 2
    for r in (a, b):
        r.pos[:] = np.clip(r.pos, [m, m], [config.WORLD_W - m, config.WORLD_H - m])
    return True


class Rival:
    def __init__(self, level, rng):
        self.level = level
        self.p = config.RIVAL_LEVELS[level]
        self.rng = rng
        self.robot = Robot(config.RIVAL_START, -math.pi / 2, self.p["speed"])
        self.goal_pos = np.asarray(config.RIVAL_GOAL_POS, dtype=np.float64)
        self.cortex = CortexSim(self.p["neurons"], rng, self.p["noise"], config.GAME_NOISE_TYPE)
        self.calib_info = self._calibrate()
        self.reset()

    def reset(self):
        self.robot.reset()
        self.grip_switch = GripSwitch()
        self.intent_v = np.zeros(2)        # 動かしたい手の速度（擬似ニューロンへの入力）
        self.intent_g = 0.0                # 握りたいか（擬似ニューロンへの入力）
        self.state, self.timer = "think", self.p["think"]
        self.target = None
        self.held = None
        self.final_goal = None             # よけて進むときの、本来の行き先
        self.reach_t = 0.0
        self.prog_pos, self.prog_t = self.robot.pos.copy(), 0.0   # 進めているか（詰まったら行き先を変える）
        self.skip_ids = set()              # 詰まって行けなかった物（しばらく狙わない）
        self.wait = 0.0                    # ためらい（強さの pause）の残り時間
        self.retry = 0
        self.score = self.collected = self.failures = self.combo = 0
        self.steal_tries = self.steals = 0
        self.frozen_t = 0.0                # 止まっている残り時間
        self.cooldown_t = 0.0              # 次に止められるまでの残り時間
        self.freezes = 0
        self.events = []                   # ログ用（main.py が毎フレーム取り出す）

    # --------------------------------------------------------
    # ライバル専用の擬似 BCI
    # --------------------------------------------------------
    def _calibrate(self):
        """同期フェーズを画面に出さずに模擬して、ライバル専用のデコーダを学習する（ライバルの手は課題のとおりに動く）"""
        hc = HandCalibration()
        hc.start()
        while hc.active:
            v = target_velocity(hc.t) + self.rng.normal(0, 0.02, 2)
            g = grip_target(hc.t)
            self.cortex.step(v, g, True)
            hc.step(self.cortex.rate_hat, self.cortex.grip_rate_hat, v, g, True)
        return self.cortex.fit(*hc.data())

    def neural_step(self):
        """20 ms ごと（main.py のシミュレーションの刻み）。止まっている間はニューロンも止める"""
        if self.frozen_t <= 0:
            self.cortex.step(self.intent_v, self.intent_g, True)

    # --------------------------------------------------------
    @property
    def combo_mult(self):
        return config.COMBO_LEVELS[self.combo]

    @property
    def frozen(self):
        return self.frozen_t > 0

    @property
    def can_freeze(self):
        return self.frozen_t <= 0 and self.cooldown_t <= 0

    @property
    def status(self):
        """HUD・ログ用：(状態, 残り時間)。"frozen" / "cooldown" / "ready"（止められる）"""
        if self.frozen_t > 0:
            return "frozen", self.frozen_t
        if self.cooldown_t > 0:
            return "cooldown", self.cooldown_t
        return "ready", 0.0

    def freeze(self):
        if not self.can_freeze:
            return False
        self.frozen_t = config.FREEZE_SEC
        self.cooldown_t = config.FREEZE_COOLDOWN
        self.freezes += 1
        self.robot.stop()
        self.intent_v[:] = 0.0
        self.events.append("freeze")
        return True

    def _log(self, text):
        self.events.append(text)

    # --------------------------------------------------------
    # 行動
    # --------------------------------------------------------
    def _value(self, it, player):
        """狙う価値：点数 ÷ かかる距離。プレイヤーのロボットのすぐそばの物は避ける"""
        pts, _ = item_points(SimpleNamespace(pickup_pos=it.pos, spec=it.spec), self.goal_pos, self.combo_mult)
        dist = np.linalg.norm(it.pos - self.robot.pos) + np.linalg.norm(it.pos - self.goal_pos) + 0.2
        if np.linalg.norm(it.pos - player.pos) < 0.13:
            pts *= 0.2
        return pts / dist

    def _go_to(self, pos):
        """台座を pos の手前（アームが届く距離）へ"""
        self.robot.approach(pos)
        self.robot.offset = None
        self.final_goal = None

    def _choose(self, field, player, player_target):
        cand = [it for it in field.selectable() if it.id not in self.skip_ids] or field.selectable()
        self.skip_ids.clear()
        if not cand:
            return None
        if (player_target is not None and player_target in cand and self._can_race(player_target, player)
                and self.rng.random() < self.p["steal"]):
            self.steal_tries += 1
            self._log(f"rival_steal_try:{player_target.id}")
            return player_target
        return max(cand, key=lambda it: self._value(it, player))

    def _can_race(self, item, player):
        """横取りしにいく意味があるか：プレイヤーより大きく遠くなければ（勝ち目のない横取りはしない）"""
        return np.linalg.norm(item.pos - self.robot.pos) <= np.linalg.norm(item.pos - player.pos) + 0.15

    def on_player_lock(self, item, player):
        """プレイヤーが物にロックオンしたとき：強さに応じた確率で、その物を横取りしにいく"""
        if self.held is not None or self.frozen or item is None or item is self.target:
            return
        if (self.state in ("think", "go", "reach") and self._can_race(item, player)
                and self.rng.random() < self.p["steal"]):
            self.steal_tries += 1
            self._log(f"rival_steal_try:{item.id}")
            self.target = item
            self._go_to(item.pos)
            self.state = "go"

    def _arm_toward(self, aim, dt):
        """手先を aim へ近づけたい速度を、擬似ニューロンへの入力（手の速度）にする。アームはデコード結果で動く"""
        tip = gripper_tip(self.robot)
        err = np.asarray(aim) - tip
        vw = err * config.RIVAL_ARM_GAIN
        sp = float(np.linalg.norm(vw))
        if sp > self.p["arm"]:
            vw *= self.p["arm"] / sp
        self.intent_v = vw / config.ARM_VEL_GAIN
        return float(np.linalg.norm(err))

    def _avoid(self, player):
        """プレイヤーのロボットが行く手のすぐ前にいたら、横へ寄り道する（簡単な衝突回避）"""
        r = self.robot
        if r.goal is None:
            return
        to_goal = r.goal - r.pos
        to_p = player.pos - r.pos
        dg, dp = float(np.linalg.norm(to_goal)), float(np.linalg.norm(to_p))
        if dp > 0.24 or dg < 1e-6 or dp > dg + 0.05:
            if self.final_goal is not None and np.linalg.norm(r.goal - r.pos) < 0.02:
                r.goal = self.final_goal            # 寄り道が終わったら本来の行き先へ
                self.final_goal = None
            return
        cosang = float(np.dot(to_goal, to_p)) / (dg * dp)
        if cosang > 0.75 and self.final_goal is None:
            side = np.array([-to_p[1], to_p[0]]) / dp
            if np.dot(side, to_goal) < 0:
                side = -side
            self.final_goal = r.goal.copy()
            m = 0.06
            r.goal = np.clip(r.pos + side * 0.16 + to_p / dp * 0.05, [m, m], [config.WORLD_W - m, config.WORLD_H - m])

    def update(self, dt, field, player, player_target):
        """
        毎フレーム。player：プレイヤーの Robot、player_target：プレイヤーがロックオンしている物（なければ None）。
        戻り値：つかんだ物（このフレームでつかんだら）、または None
        """
        grabbed = None
        if self.frozen_t > 0:
            self.frozen_t = max(0.0, self.frozen_t - dt)
            self.robot.stop()
            if self.frozen_t == 0:
                self._log("unfreeze")
                if self.state == "go" and self.target is not None:
                    self._go_to(self.target.pos)
                elif self.state == "carry":
                    self._go_to(self.goal_pos)
            if self.held is not None:
                self.held.pos = gripper_tip(self.robot)
            return None
        self.cooldown_t = max(0.0, self.cooldown_t - dt)

        r = self.robot
        # 狙っている物が、取られた・つかまれた・消えたら考え直す
        if self.state in ("go", "reach", "grip") and (self.target is None or self.target not in field.items
                                                       or not self.target.landed):
            self.target = None
            self.state, self.timer = "think", self.p["think"] * 0.5
            self.intent_g = 0.0

        # 詰まっていないか：台座が動こうとしているのに 1.2 秒進めなければ、物なら狙いを変え、運んでいるなら横へよける
        if r.moving and self.state in ("go", "carry"):
            if np.linalg.norm(r.pos - self.prog_pos) > 0.01:
                self.prog_pos, self.prog_t = r.pos.copy(), 0.0
            else:
                self.prog_t += dt
                if self.prog_t > 1.2:
                    self.prog_t = 0.0
                    self._log("rival_stuck")
                    if self.state == "go" and self.target is not None:
                        self.skip_ids.add(self.target.id)
                        self.target, self.state, self.timer = None, "think", 0.3
                        r.stop()
                    else:
                        side = self.rng.choice([-1.0, 1.0])
                        d = (r.goal - r.pos) if r.goal is not None else np.array([1.0, 0.0])
                        n = d / max(np.linalg.norm(d), 1e-6)
                        m = 0.06
                        self.final_goal = self.goal_pos.copy()
                        r.goal = np.clip(r.pos + side * np.array([-n[1], n[0]]) * 0.18, [m, m],
                                         [config.WORLD_W - m, config.WORLD_H - m])
        else:
            self.prog_pos, self.prog_t = r.pos.copy(), 0.0

        self.intent_v = np.zeros(2)
        if self.wait > 0:                              # ためらい中：何もしない（アームは待機の位置へ戻っていく）
            self.wait -= dt
        elif self.state == "think":
            self.timer -= dt
            if self.timer <= 0:
                self.target = self._choose(field, player, player_target)
                if self.target is None:
                    self.timer = 0.5
                else:
                    self.retry = 0
                    self._go_to(self.target.pos)
                    self.state = "go"
                    self._log(f"rival_lock:{self.target.id}")
        elif self.state == "go":
            self._avoid(player)
            if not r.moving:
                self.state, self.reach_t, self.wait = "reach", 0.0, self.p["pause"]
        elif self.state == "reach":
            self.reach_t += dt
            d = self._arm_toward(self.target.pos, dt)
            if d < self.target.radius * config.RIVAL_GRIP_REACH or self.reach_t > config.RIVAL_REACH_TIMEOUT:
                self.intent_g, self.state, self.reach_t = 1.0, "grip", 0.0
        elif self.state == "grip":
            self.reach_t += dt
            self._arm_toward(self.target.pos, dt)
            if self.reach_t > 2.5:                     # 握れないまま時間切れ：いったん開いてやり直す
                self.intent_g, self.state, self.reach_t = 0.0, "reach", 0.0
        elif self.state == "release":                  # つかみ損ねたあと、手を開くのを待つ
            if not self.grip_switch.closed:
                if self.retry > config.RIVAL_MAX_RETRY:
                    self.state, self.timer = "think", self.p["think"]
                else:
                    self.state, self.reach_t = "reach", 0.0
        elif self.state == "carry":
            self._avoid(player)
            if not r.moving:
                self.state, self.reach_t, self.wait = "place", 0.0, self.p["pause"] * 0.5
        elif self.state == "place":
            self.reach_t += dt
            d = self._arm_toward(self.goal_pos, dt)
            if d < config.GOAL_RADIUS * 0.45 or self.reach_t > config.RIVAL_REACH_TIMEOUT:
                self.intent_g = 0.0

        # アーム：台座が止まっているときだけ、デコードした速度で動かす（プレイヤーと同じ）
        if not r.moving:
            r.drive_arm(self.cortex.v_hat * config.ARM_VEL_GAIN, dt)

        # グリッパー：デコードした把持の値で開閉する（プレイヤーと同じヒステリシス）
        gd = self.cortex.grip
        ev = self.grip_switch.update(gd.value, gd.close_th, gd.open_th, dt)
        if ev == "close":
            r.set_grip(True)
            grabbed = self._on_close(field)
        elif ev == "open":
            r.set_grip(False)
            self._on_open(field)
        if self.held is not None:
            self.held.pos = gripper_tip(r)
        return grabbed

    def _on_close(self, field):
        if self.held is not None or self.state not in ("reach", "grip"):
            return None
        tip = gripper_tip(self.robot)
        cand = field.selectable()
        best = min(cand, key=lambda it: np.linalg.norm(it.pos - tip), default=None)
        if best is None or np.linalg.norm(best.pos - tip) > best.radius + config.GRAB_MARGIN:
            self.failures += 1
            self.retry += 1
            self.combo = max(0, self.combo - 1)
            self.intent_g = 0.0
            self.state = "release"
            self._log("rival_grasp_miss")
            return None
        best.held = True
        best.pickup_pos = best.pos.copy()
        self.held = best
        self.target = None
        self.state = "carry"
        self.wait = self.p["pause"]
        self._go_to(self.goal_pos)
        self._log(f"rival_grasp_ok:{best.type}")
        return best

    def _on_open(self, field):
        if self.held is None:
            return
        it = self.held
        tip = gripper_tip(self.robot)
        it.held = False
        it.pos = tip.copy()
        it.height = 0.08
        self.held = None
        if np.linalg.norm(tip - self.goal_pos) <= config.GOAL_RADIUS:
            pts, _ = item_points(it, self.goal_pos, self.combo_mult)
            self.score += pts
            self.collected += 1
            self.combo = min(len(config.COMBO_LEVELS) - 1, self.combo + 1)
            field.remove(it)
            field.schedule_refill()
            self._log(f"rival_deliver:{it.type}:+{pts}")
        else:
            self.failures += 1
            self.combo = max(0, self.combo - 1)
            self._log(f"rival_drop:{it.type}")
        self.state, self.timer = "think", self.p["think"]
