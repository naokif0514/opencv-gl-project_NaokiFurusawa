"""
highscore.py
------------
ハイスコア（上位 HIGHSCORE_KEEP 件）を、名前なし（日時・スコア・集めた数）で JSON に保存する。
保存先は saves/highscores.json（git には入れない）。
モードごとに分けて保存する：{"normal": [...], "challenge_weak": [...], "challenge_normal": [...], "challenge_strong": [...]}
  normal = ノーマル（一人で集める）、challenge_強さ = チャレンジ（ライバルとの対戦。ライバルの強さごと）
以前の「チャレンジ = 電極劣化あり」の記録（"challenge"、またはモードを分ける前の形式 = リストだけ）は、
ルールが違うので新しいランキングには混ぜず、最初に読んだときに saves/highscores_old_challenge.json へ移して残す
"""

import json
import os
from datetime import datetime

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, config.HIGHSCORE_FILE)


MODES = ("normal", "challenge_weak", "challenge_normal", "challenge_strong")
OLD_PATH = os.path.join(BASE_DIR, config.HIGHSCORE_OLD_CHALLENGE_FILE)


def _move_old_challenge(data):
    """以前のチャレンジ（電極劣化あり）の記録を別のファイルへ移す（すでにあれば足す）。移したら True"""
    if isinstance(data, list):                 # モードを分ける前の形式（劣化ありで遊んだ記録）
        old, data = data, {}
    elif isinstance(data, dict) and "challenge" in data:
        old = data.pop("challenge")
    else:
        return data, False
    try:
        with open(OLD_PATH, encoding="utf-8") as fp:
            prev = json.load(fp)
    except (OSError, ValueError):
        prev = []
    try:
        os.makedirs(os.path.dirname(OLD_PATH), exist_ok=True)
        with open(OLD_PATH, "w", encoding="utf-8") as fp:
            json.dump((prev if isinstance(prev, list) else []) + list(old), fp, ensure_ascii=False, indent=1)
        print(f"[highscore] 以前のチャレンジ（電極劣化あり）の記録 {len(old)} 件を {OLD_PATH} へ移しました")
    except OSError as e:
        print(f"[highscore] 以前の記録を移せませんでした: {e}")
        return data, False
    return data, True


def _load_all():
    """{モード: 記録のリスト}"""
    try:
        with open(PATH, encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError):
        data = {}
    data, moved = _move_old_challenge(data)
    if not isinstance(data, dict):
        data = {}
    out = {m: [d for d in data.get(m, []) if isinstance(d, dict) and "score" in d] for m in MODES}
    if moved:
        _save(out)
    return out


def _save(data):
    try:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=1)
    except OSError as e:
        print(f"[highscore] 保存できませんでした: {e}")


def load(mode="normal"):
    return _load_all().get(mode, [])


def add(score, collected, mode="normal"):
    """今回の結果をそのモードの記録に加えて上位だけ保存する。(並べかえたリスト, 今回の順位（入らなければ None）) を返す"""
    entry = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": int(score), "collected": int(collected)}
    data = _load_all()
    scores = data[mode] + [entry]
    scores.sort(key=lambda d: -d["score"])
    scores = scores[:config.HIGHSCORE_KEEP]
    data[mode] = scores
    rank = next((i for i, d in enumerate(scores) if d is entry), None)
    _save(data)
    return scores, rank
