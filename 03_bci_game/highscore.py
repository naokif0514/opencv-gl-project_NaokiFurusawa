"""
highscore.py
------------
ハイスコア（上位 HIGHSCORE_KEEP 件）を、名前なし（日時・スコア・集めた数）で JSON に保存する。
保存先は saves/highscores.json（git には入れない）。
"""

import json
import os
from datetime import datetime

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, config.HIGHSCORE_FILE)


def load():
    try:
        with open(PATH, encoding="utf-8") as fp:
            data = json.load(fp)
        return [d for d in data if isinstance(d, dict) and "score" in d]
    except (OSError, ValueError):
        return []


def add(score, collected):
    """今回の結果を加えて上位だけ保存する。(並べかえたリスト, 今回の順位（入らなければ None）) を返す"""
    entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": int(score), "collected": int(collected)}
    scores = load() + [entry]
    scores.sort(key=lambda d: -d["score"])
    scores = scores[:config.HIGHSCORE_KEEP]
    rank = next((i for i, d in enumerate(scores) if d is entry), None)
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as fp:
            json.dump(scores, fp, ensure_ascii=False, indent=1)
    except OSError as e:
        print(f"[highscore] 保存できませんでした: {e}")
    return scores, rank
