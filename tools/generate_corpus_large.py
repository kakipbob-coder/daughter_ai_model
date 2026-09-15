import os
import random

OUTPUT_PATH = "data/train.txt"

# 文体カテゴリ
general_topics = [
    "日常生活の描写",
    "技術的な説明",
    "自然現象の描写",
    "抽象概念の説明",
    "物語の一部",
    "歴史的な出来事の解説",
    "心理描写",
    "社会問題の考察",
    "文化的な習慣の説明",
    "科学的な現象の説明",
]

story_templates = [
    "ある日、{place}で{character}は{event}。その出来事は小さな変化のようでいて、後に大きな意味を持つことになる。",
    "{character}は{place}を歩きながら、{emotion}気持ちを抱えていた。周囲の景色は変わらないようでいて、どこか違って見えた。",
    "{place}に住む人々は、{event}という噂を耳にして落ち着かない日々を過ごしていた。",
]

technical_templates = [
    "{topic}は現代の情報処理において重要な概念である。基本的には{explain}という仕組みで動作する。",
    "システム設計では{topic}が中心的な役割を果たす。これは{explain}ために必要となる。",
]

sns_templates = [
    "今日は{emotion}気分。なんかうまくいかないけど、まあいいか。",
    "A: {question}\nB: {answer}",
    "最近さ、{complaint}って思うんだよね。",
    "正直、{emotion}けど頑張るしかないよな。",
]

places = ["静かな町", "大都市", "海辺の村", "山間の集落", "古い商店街"]
characters = ["青年", "少女", "老人", "旅人", "研究者"]
events = ["奇妙な光を見た", "不思議な手紙を受け取った", "小さな違和感を覚えた", "突然の知らせを聞いた"]
emotions = ["不安な", "嬉しい", "落ち着かない", "ワクワクする", "複雑な"]
topics = ["データベース", "ニューラルネットワーク", "通信プロトコル", "暗号化技術"]
explains = ["情報を整理する", "効率的に処理する", "安全に伝達する"]
questions = ["今どこ？", "今日どうする？", "もう着いた？"]
answers = ["駅にいるよ", "これから向かう", "あと5分で着く"]
complaints = ["天気が安定しない", "仕事が終わらない", "眠気が取れない"]


def gen_general():
    topic = random.choice(general_topics)
    return f"{topic}についての説明文です。これは自動生成された文章であり、特定の著作物には依存していません。内容は一般的で、文法的に自然な日本語となるように構成されています。"


def gen_story():
    return random.choice(story_templates).format(
        place=random.choice(places),
        character=random.choice(characters),
        event=random.choice(events),
        emotion=random.choice(emotions),
    )


def gen_technical():
    return random.choice(technical_templates).format(
        topic=random.choice(topics),
        explain=random.choice(explains),
    )


def gen_sns():
    return random.choice(sns_templates).format(
        emotion=random.choice(emotions),
        question=random.choice(questions),
        answer=random.choice(answers),
        complaint=random.choice(complaints),
    )


def main():
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # 30MB級生成（約30万行）
        for _ in range(200000):
            f.write(gen_general() + "\n")
            f.write(gen_story() + "\n")
            f.write(gen_technical() + "\n")

        # SNS風（少量）
        for _ in range(20000):
            f.write(gen_sns() + "\n")

    print(f"Generated large corpus: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
