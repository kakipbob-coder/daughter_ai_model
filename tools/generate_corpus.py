import os
import random

OUTPUT_PATH = "data/train.txt"

# 文章生成のためのテンプレート
general_topics = [
    "日常生活の話題",
    "技術的な説明",
    "自然現象の描写",
    "抽象的な概念の説明",
    "物語の一部",
    "歴史的な出来事の解説",
    "心理描写",
    "社会問題の考察",
]

sns_styles = [
    "X風の短文",
    "LINE風の会話",
    "雑談",
    "軽い愚痴",
    "ネットスラングを含む文章",
]

def generate_general_text():
    topic = random.choice(general_topics)
    return f"{topic}についての説明文です。これは自動生成された文章であり、特定の著作物には依存していません。内容は一般的で、文法的に自然な日本語となるように構成されています。"

def generate_story_text():
    return "ある日、静かな町に小さな変化が訪れた。人々はその違和感に気づきながらも、日常を続けていた。これは物語の一部として自動生成された文章である。"

def generate_technical_text():
    return "コンピュータシステムは入力、処理、出力の三つの要素で構成される。これらは情報処理の基本概念であり、自動生成された説明文である。"

def generate_sns_text():
    style = random.choice(sns_styles)
    if style == "X風の短文":
        return "今日はなんか集中できないけど、とりあえず頑張る。"
    if style == "LINE風の会話":
        return "A: 今どこ？\nB: 駅に着いたよ。"
    if style == "雑談":
        return "最近さ、天気が安定しなくて困るんだよね。"
    if style == "軽い愚痴":
        return "仕事が終わらん…コーヒー飲んで気合い入れるか。"
    return "なんか草。"

def main():
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 70% 一般文
        for _ in range(7000):
            f.write(generate_general_text() + "\n")

        # 20% 技術・物語
        for _ in range(2000):
            f.write(generate_technical_text() + "\n")
            f.write(generate_story_text() + "\n")

        # 10% SNS風
        for _ in range(1000):
            f.write(generate_sns_text() + "\n")

    print(f"Generated: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
