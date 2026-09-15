# -*- coding: utf-8 -*-
"""
JIS X 0208 の全文字（kanji.txt）を利用した
高多様性コーパス自動生成スクリプト（完全版）

出力: data/train.txt
"""

import os
import random

OUTPUT_PATH = "data/train.txt"
KANJI_PATH = "data/kanji.txt"

# -----------------------------
# ① JIS X 0208 全文字を読み込む
# -----------------------------
with open(KANJI_PATH, "r", encoding="utf-8") as f:
    KANJI = list(f.read().strip())

# -----------------------------
# ② 日本語語彙生成用の基本セット
# -----------------------------
HIRAGANA = list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
KATAKANA = list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")

YOJIJUKUGO = [
    "一期一会", "温故知新", "起死回生", "電光石火", "森羅万象",
    "百花繚乱", "虚心坦懐", "臥薪嘗胆", "大胆不敵", "完全無欠",
    "深謀遠慮", "自由自在", "千変万化", "不言実行", "質実剛健",
]

# -----------------------------
# ③ 語彙生成関数
# -----------------------------
def random_kanji_word(min_len=2, max_len=6):
    return "".join(random.choice(KANJI) for _ in range(random.randint(min_len, max_len)))

def random_hiragana_word():
    return "".join(random.choice(HIRAGANA) for _ in range(random.randint(3, 8)))

def random_katakana_word():
    return "".join(random.choice(KATAKANA) for _ in range(random.randint(3, 8)))

def random_mixed_word():
    return random.choice([
        random_kanji_word(),
        random_hiragana_word(),
        random_katakana_word(),
    ])

# -----------------------------
# ④ 文生成テンプレート
# -----------------------------
def gen_diverse_sentence():
    w1 = random_mixed_word()
    w2 = random_mixed_word()
    w3 = random_mixed_word()
    return f"{w1}という概念は、{w2}の文脈において{w3}と関連付けられることがある。これは自動生成された文章であり、特定の著作物には依存しない。"

def gen_story():
    return f"{random_kanji_word()}の町で{random_kanji_word()}が{random_kanji_word()}を見つけた。その出来事は{random_kanji_word()}の始まりだった。"

def gen_technical():
    return f"{random_kanji_word()}システムは{random_kanji_word()}処理を行うために設計されている。基本構造は{random_kanji_word()}モデルに基づく。"

def gen_yojijukugo():
    y = random.choice(YOJIJUKUGO)
    return f"{y}という表現は、状況を端的に示すために用いられることがある。"

def gen_sns():
    return f"今日は{random_hiragana_word()}感じ。{random_katakana_word()}だけど頑張る。"

# -----------------------------
# ⑤ メイン処理（50MB級生成）
# -----------------------------
def main():
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 約 1,000,000 行 → 50MB級
        for _ in range(800000):
            f.write(gen_diverse_sentence() + "\n")
            f.write(gen_story() + "\n")
            f.write(gen_technical() + "\n")
            f.write(gen_yojijukugo() + "\n")

        # SNS風（少量）
        for _ in range(20000):
            f.write(gen_sns() + "\n")

    print(f"Generated large diverse corpus: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
