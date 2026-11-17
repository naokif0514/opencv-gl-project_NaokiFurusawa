"""
highscore.py
------------
ハイスコア（上位 HIGHSCORE_KEEP 件）を、名前なし（日時・スコア・集めた数）で JSON に保存する。
保存先は saves/highscores.json（git には入れない）。
モード（"normal" = 電極劣化なし、"challenge" = 劣化あり）ごとに分けて保存する：{"normal": [...], "challenge": [...]}
（モードを分ける前の形式（リストだけ）は、劣化ありで遊んだ記録なので "challenge" として読む）
"""

import json
import os
from datetime import datetime

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, config.HIGHSCORE_FILE)


MODES = ("normal", "challenge")


def _load_all():
    """{モード: 記録のリスト}"""
    try:
        with open(PATH, encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        data = {}
    if isinstance(data, list):                 # モードを分ける前の形式
        data = {"challenge": data}
    if not isinstance(data, dict):
        data = {}
    return {m: [d for d in data.get(m, []) if isinstance(d, dict) and "score" in d] for m in MODES}


def load(mode="challenge"):
    return _load_all().get(mode, [])


def add(score, collected, mode="challenge"):
    """今回の結果をそのモードの記録に加えて上位だけ保存する。(並べかえたリスト, 今回の順位（入らなければ None）) を返す"""
    entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": int(score), "collected": int(collected)}
    data = _load_all()
    scores = data[mode] + [entry]
    scores.sort(key=lambda d: -d["score"])
    scores = scores[:config.HIGHSCORE_KEEP]
    data[mode] = scores
    rank = next((i for i, d in enumerate(scores) if d is entry), None)
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=1)
    except OSError as e:
        print(f"[highscore] 保存できませんでした: {e}")
    return scores, rank
