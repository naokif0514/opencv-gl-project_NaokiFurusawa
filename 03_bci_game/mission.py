"""
mission.py
----------
ゲームのルール（1 ステージ制）：制限時間内に、机の上の物をどれだけ集められるか。

■ 1 個を集める流れ
  1. 物を見つめてロックオン → ロボットが近くへ移動（視線）
  2. 手を動かしてアームを伸ばし（運動野の発火 → 速度のデコード）、手を握ってつかむ（把持ニューロン → 把持のデコード）
  3. カゴを見つめてロボットを移動させ、手を開いてカゴに置く → 得点
  一度に運べるのは 1 個。何もないところで握る（つかみ損ね）・カゴの外で放す（床に落ちて残る）と失敗。

■ 得点
  基本点（物の種類。小さい物ほど高い。距離では変えない）
  × コンボの倍率（COMBO_LEVELS。成功で 1 段階上がり、失敗で 1 段階だけ下がる）

■ 電極劣化イベント
  ランダムな時刻（DEGRADE_WINDOW の範囲）に 1〜2 回（DEGRADE_COUNT）、2 回なら DEGRADE_MIN_GAP 秒以上空けて起きる。
  方向のニューロンの DEGRADE_FRACTION、把持ニューロンの DEGRADE_GRIP_FRACTION が脱落する。デコーダはそれを知らないので、
  推定速度にかたよりが乗り、アームが一方へ流される。L で再同期すると時間は使うが精度が戻る。
  （把持ニューロンを同じ割合で脱落させると、握っても「握った」のしきい値に届かず、再同期しないと
    まったく遊べなくなるので少なめにしている）

■ ボーナス
  現実の色の物を見せると降ってくる。1 回降ったら BONUS_COOLDOWN 秒は降らない。1 ゲームで BONUS_MAX 回まで。
"""

import math
from dataclasses import dataclass, field

import numpy as np

import config


class GripSwitch:
    """
    把持の推定値（0〜1、なめらかにしたもの）を、ヒステリシスつきのしきい値で「握った / 開いた」に変える。
    閉じるしきい値と開くしきい値を変えておくと、しきい値の近くで推定値が揺れても、グリッパーがパタパタしない
    """

    def __init__(self):
        # 最初は「閉じている」扱いにして、一度手を開いた（推定値が下のしきい値を下回った）あとで
        # 「握る」を受け付ける（同期フェーズは握る区間で終わるので、直後に勝手に握ったことにならないように）
        self.closed = True
        self.below = 0.0            # 推定値が「開いた」のしきい値の下に続いている時間 [秒]

    def update(self, value, close_th=config.GRIP_CLOSE_TH, open_th=config.GRIP_OPEN_TH, dt=0.0,
               hold=config.GRIP_OPEN_HOLD_SEC):
        """
        状態が変わったら "close" / "open" を返す（しきい値は把持のデコーダが同期フェーズで決めたもの）。
        開くのは、推定値が open_th 以下に hold 秒続いたとき（把持ニューロンの推定値は、手を握ったままでも
        ノイズで一瞬しきい値を下回ることがあり、そのたびに物を落としていたため）
        """
        if not self.closed:
            if value >= close_th:
                self.closed = True
                self.below = 0.0
                return "close"
            return None
        if value <= open_th:
            self.below += dt
            if self.below >= hold:
                self.closed = False
                self.below = 0.0
                return "open"
        else:
            self.below = 0.0
        return None


def gripper_tip(robot):
    """グリッパーの先（手首の少し先）"""
    pose = robot.pose
    a = pose.theta1 + pose.theta2
    return pose.hand + 0.022 * np.array([math.cos(a), math.sin(a)])


def degrade_schedule(rng):
    """電極劣化の時刻（プレイ開始からの秒）。2 回なら DEGRADE_MIN_GAP 秒以上空ける"""
    lo, hi = config.DEGRADE_WINDOW
    n = int(rng.integers(config.DEGRADE_COUNT[0], config.DEGRADE_COUNT[1] + 1))
    for _ in range(200):
        ts = np.sort(rng.uniform(lo, hi, n))
        if n < 2 or np.min(np.diff(ts)) >= config.DEGRADE_MIN_GAP:
            return [float(t) for t in ts]
    return [float(lo + i * config.DEGRADE_MIN_GAP) for i in range(n)]


@dataclass
class GameResult:
    score: int = 0
    collected: int = 0
    failures: int = 0
    max_combo: float = 1.0
    by_type: dict = field(default_factory=dict)
    events: list = field(default_factory=list)       # (プレイ時間, 文字列)
    knocked: int = 0                                 # チャレンジ：ライバルにぶつかられて落とした回数


def item_points(item, goal, combo_mult):
    """物をカゴに入れたときの点数：物の種類の基本点 × コンボの倍率（つかんだ位置の距離では変えない）。goal は使わない（以前の名残）"""
    return int(round(item.spec["points"] * combo_mult))


class Mission:
    def __init__(self, rng, degrade=True):
        self.rng = rng
        self.degrade_enabled = degrade    # False（ノーマル）なら電極劣化は起きない
        self.result = GameResult()
        self.time_left = config.GAME_TIME
        self.play_time = 0.0
        self.held = None                  # つかんでいる物（FieldItem）
        self.combo = 0                    # COMBO_LEVELS の段階
        self.degrade_times = degrade_schedule(rng) if degrade else []
        self.degrade_count = 0
        self.bonus_count = 0
        self.last_bonus = -1e9
        self.last_score_delta = 0

    @property
    def done(self):
        return self.time_left <= 0

    @property
    def combo_mult(self):
        return config.COMBO_LEVELS[self.combo]

    def log(self, text):
        self.result.events.append((round(self.play_time, 2), text))

    def tick(self, dt):
        self.play_time += dt
        self.time_left = max(0.0, self.time_left - dt)

    def degrade_due(self):
        """電極劣化の時刻になったら True（1 回につき 1 度だけ）"""
        if self.degrade_count < len(self.degrade_times) and self.play_time >= self.degrade_times[self.degrade_count]:
            self.degrade_count += 1
            return True
        return False

    def bonus_allowed(self):
        return (self.bonus_count < config.BONUS_MAX
                and self.play_time - self.last_bonus >= config.BONUS_COOLDOWN)

    def on_bonus(self):
        self.bonus_count += 1
        self.last_bonus = self.play_time
        self.log("bonus_drop")

    # --------------------------------------------------------
    def _fail(self, what):
        self.result.failures += 1
        self.combo = max(0, self.combo - 1)          # 失敗したらコンボを 1 段階だけ下げる
        self.log(what)

    def on_close(self, tip, items):
        """グリッパーを閉じたとき：先端の近くに物があればつかむ。なければつかみ損ね"""
        if self.held is not None:
            return None
        cand = [it for it in items if it.landed]
        best = min(cand, key=lambda it: np.linalg.norm(it.pos - tip), default=None)
        if best is None or np.linalg.norm(best.pos - tip) > best.radius + config.GRAB_MARGIN:
            self._fail("grasp_miss")
            return "miss"
        best.held = True
        best.pickup_pos = best.pos.copy()
        self.held = best
        self.log(f"grasp_ok:{best.type}")
        return best

    def points_for(self, item):
        """この物をカゴに入れたときの点数：基本点 × コンボの倍率"""
        return item_points(item, config.GOAL_POS, self.combo_mult)

    def knock_off(self, tip):
        """チャレンジ：運んでいる途中でライバルにぶつかられて、物を落とす（床に残る。コンボは下げない）"""
        if self.held is None:
            return None
        it = self.held
        it.held = False
        it.pos = tip.copy()
        it.height = 0.08
        self.held = None
        self.result.knocked += 1
        self.log(f"knocked:{it.type}")
        return it

    def on_open(self, tip):
        """グリッパーを開いたとき：カゴの中なら得点、外なら床に落ちて残る（失敗）"""
        if self.held is None:
            return None
        it = self.held
        it.held = False
        it.pos = tip.copy()
        it.height = 0.08                 # 少し上から床に落ちる
        self.held = None
        d = float(np.linalg.norm(tip - np.asarray(config.GOAL_POS)))
        if d <= config.GOAL_RADIUS:
            pts = self.points_for(it)
            self.result.score += pts
            self.result.collected += 1
            self.result.by_type[it.type] = self.result.by_type.get(it.type, 0) + 1
            self.last_score_delta = pts
            self.log(f"deliver_ok:{it.type}:+{pts}:combo_x{self.combo_mult:.1f}")
            self.combo = min(len(config.COMBO_LEVELS) - 1, self.combo + 1)
            self.result.max_combo = max(self.result.max_combo, self.combo_mult)
            return ("deliver", it, pts)
        self._fail(f"drop:{it.type}")
        return ("drop", it, 0)
