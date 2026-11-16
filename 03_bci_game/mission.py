"""
mission.py
----------
ステージ（難易度）とミッション（アイテムをつかんでゴールに置く）の進行、スコア。

■ 1 回のミッションの流れ
  1. アイテムを見つめてロックオン → ロボットが近くへ移動（視線：前頭眼野・視覚野）
  2. 手を動かしてアームを伸ばし（運動野の発火 → 速度のデコード）、手を握ってつかむ（把持ニューロン → 把持のデコード）
  3. ゴールを見つめてロボットを移動させ、手を開いて置く
  4. ゴールの中に置けたら成功。ゴールからの距離が近いほど高得点。
     何もないところで握る（つかみ損ね）・ゴールの外で放す（落とす）と失敗 1 回

■ スコア
  置けたとき     : DELIVERY_SCORE + ACCURACY_SCORE × (1 − ゴールの中心からの距離 / ゴールの半径)
  失敗 1 回ごと   : − FAIL_PENALTY
  ステージクリア : 残り時間 × TIME_SCORE（秒あたり）

■ 電極劣化イベント（ステージ 3）
  方向のニューロンの 25%、把持ニューロンの 5% が脱落する。デコーダはそれを知らないので、推定速度に一定の向きの
  かたよりが乗り、アームが一方へ流される。模擬プレイヤーで調べると（8 秒以内につかめた割合）、
    再同期しない：6/12（流される向きと逆へ手を動かして補えば、なんとかつかめる）
    再同期する  ：12/12（約 0.7 秒でつかめる）
  だったので、この割合にした（把持ニューロンも同じ割合で脱落させると、握っても「握った」のしきい値に届かず、
  再同期しないとまったく遊べなくなる）。再同期（約 21 秒）は制限時間を使うので、するかどうかをプレイヤーが判断する。

■ 現実の物（色の物）のとき
  現実の物は動かせないので、つかんだときに「スキャン」してデジタルの荷物（同じ色の立方体）を作り、それを運ぶ。
  スキャンした物は、その場所から動かすまでは、もう一度は選べない。
"""

import math
from dataclasses import dataclass, field

import numpy as np

import config
from items import Item, _ids


@dataclass
class StageDef:
    number: int
    title_ja: str
    title_en: str
    note_ja: str
    n_neurons: int
    noise: float
    noise_type: str
    goal_radius: float
    degrade_at: float | None = None      # プレイ開始から何秒後に電極劣化イベントを起こすか
    degrade_fraction: float = 0.0


STAGES = [
    StageDef(1, "ステージ 1：接続テスト", "STAGE 1: link test",
             "ニューロン 200 個・ノイズ少なめ。まずは操作に慣れよう", 200, 5.0, "drift", 0.075),
    StageDef(2, "ステージ 2：信号が弱い", "STAGE 2: weak signal",
             "記録できるニューロンが 60 個に減った。デコードが揺れる", 60, 5.0, "drift", 0.075),
    StageDef(3, "ステージ 3：電極劣化", "STAGE 3: electrode failure",
             "プレイ中に電極が劣化してニューロンが脱落する。再同期（L）するかは自分で判断", 150, 5.0, "drift",
             0.075, degrade_at=config.DEGRADE_AT_SEC, degrade_fraction=config.DEGRADE_FRACTION),
    StageDef(4, "ステージ 4：精密作業", "STAGE 4: precision",
             "ノイズが強い中で、小さなゴールに正確に置く", 150, 25.0, "both", 0.04),
]


class GripSwitch:
    """
    把持の推定値（0〜1、なめらかにしたもの）を、ヒステリシスつきのしきい値で「握った / 開いた」に変える。
    閉じるしきい値と開くしきい値を変えておくと、しきい値の近くで推定値が揺れても、グリッパーがパタパタしない
    """

    def __init__(self):
        # 最初は「閉じている」扱いにして、一度手を開いた（推定値が下のしきい値を下回った）あとで
        # 「握る」を受け付ける（同期フェーズは握る区間で終わるので、直後に勝手に握ったことにならないように）
        self.closed = True

    def update(self, value, close_th=config.GRIP_CLOSE_TH, open_th=config.GRIP_OPEN_TH):
        """状態が変わったら "close" / "open" を返す（しきい値は把持のデコーダが同期フェーズで決めたもの）"""
        if not self.closed and value >= close_th:
            self.closed = True
            return "close"
        if self.closed and value <= open_th:
            self.closed = False
            return "open"
        return None


@dataclass
class StageResult:
    stage: int
    score: int = 0
    delivered: int = 0
    failures: int = 0
    time_left: float = 0.0
    cleared: bool = False
    accuracy: list = field(default_factory=list)     # (プレイ時間, 相関, RMSE)：デコード精度の推移
    events: list = field(default_factory=list)


class Mission:
    def __init__(self, stage: StageDef):
        self.stage = stage
        self.result = StageResult(stage.number)
        self.time_left = config.STAGE_TIME
        self.play_time = 0.0
        self.held = None                 # つかんでいる荷物（Item）
        self.scanned = []                # スキャン済みの現実の物の位置
        self.degraded = False
        self.last_event = ""

    @property
    def done(self):
        return self.result.delivered >= config.ITEMS_PER_STAGE or self.time_left <= 0

    def log(self, text):
        self.last_event = text
        self.result.events.append((round(self.play_time, 2), text))

    def tick(self, dt):
        self.play_time += dt
        self.time_left = max(0.0, self.time_left - dt)

    def selectable(self, item):
        """視線で選べるアイテムか（つかんでいる物・スキャン済みの現実の物は選べない）"""
        if item.delivered or (self.held is not None and item.id == self.held.id):
            return False
        if item.kind == "real":
            return all(np.linalg.norm(item.pos - p) > config.SCAN_RADIUS for p in self.scanned)
        return True

    # --------------------------------------------------------
    def on_close(self, tip, items):
        """グリッパーを閉じたとき：先端の近くにアイテムがあればつかむ。なければつかみ損ね"""
        if self.held is not None:
            return None
        cand = [it for it in items if self.selectable(it)]
        best = min(cand, key=lambda it: np.linalg.norm(it.pos - tip), default=None)
        if best is None or np.linalg.norm(best.pos - tip) > best.radius + config.GRAB_MARGIN:
            self.result.failures += 1
            self.result.score -= config.FAIL_PENALTY
            self.log("grasp_miss")
            return "miss"
        if best.kind == "real":
            # 現実の物はスキャンして、同じ色のデジタルの荷物を作る
            self.scanned.append(best.pos.copy())
            best = Item(next(_ids), "virtual", best.color, tip.copy(), 0.03, angle=0.3)
            best.scanned = True
            items_add = best
        else:
            items_add = None
        best.held = True
        self.held = best
        self.log(f"grasp_ok:{best.color}")
        return ("grab", items_add)

    def on_open(self, tip, goal):
        """グリッパーを開いたとき：ゴールの中なら成功、外なら落とす（失敗）"""
        if self.held is None:
            return None
        it = self.held
        it.held = False
        it.pos = tip.copy()
        self.held = None
        d = float(np.linalg.norm(tip - goal.pos))
        if d <= goal.radius:
            acc = max(0.0, 1.0 - d / goal.radius)
            pts = int(config.DELIVERY_SCORE + config.ACCURACY_SCORE * acc)
            self.result.score += pts
            self.result.delivered += 1
            it.delivered = True
            self.log(f"deliver_ok:dist={d:.3f}:+{pts}")
            return ("deliver", it, pts, d)
        self.result.failures += 1
        self.result.score -= config.FAIL_PENALTY
        self.log(f"drop:dist={d:.3f}")
        return ("drop", it, 0, d)

    def finish(self):
        r = self.result
        r.time_left = self.time_left
        r.cleared = r.delivered >= config.ITEMS_PER_STAGE
        if r.cleared:
            r.score += int(self.time_left * config.TIME_SCORE)
        return r


def gripper_tip(robot):
    """グリッパーの先（手首の少し先）"""
    pose = robot.pose
    a = pose.theta1 + pose.theta2
    return pose.hand + 0.022 * np.array([math.cos(a), math.sin(a)])
