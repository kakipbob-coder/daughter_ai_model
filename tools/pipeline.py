import re
from typing import List, Tuple


# =========================
# テーマ抽出
# =========================
def extract_theme(user_text: str) -> str:
    """
    User文からテーマを抽出する。
    「〜について」「〜を知りたい」「〜を相談したい」「〜のコツ」「〜の方法」などに対応。
    抽出できなければ「相談内容」を返す。
    """
    patterns = [
        r"(.+?)について",
        r"(.+?)を知りたい",
        r"(.+?)を相談したい",
        r"(.+?)のコツ",
        r"(.+?)の方法",
    ]
    for p in patterns:
        m = re.search(p, user_text)
        if m:
            return m.group(1).strip()
    return "相談内容"


# =========================
# 補完用の文章生成
# =========================
def generate_explanation(theme: str) -> str:
    return (
        f"{theme}には、基本的な考え方がいくつかあります。"
        "まずは土台となる部分を押さえることが大切です。"
    )


def generate_examples(theme: str) -> str:
    return (
        f"{theme}ために役立つ行動として、次のようなものがあります。\n"
        "- 生活リズムを整える\n"
        "- 環境を整える\n"
        "- 無理のない範囲で継続する"
    )


def generate_summary(theme: str) -> str:
    return f"これらを意識することで、{theme}がより実践しやすくなります。"


# =========================
# Assistant返答の補完
# =========================
def complete_assistant_reply(user_text: str, assistant_intro: str) -> str:
    """
    Assistant の導入文（外部AIが生成した返答の先頭部分）に対して、
    説明・例示・まとめを追加して「完成した返答」にする。
    """
    theme = extract_theme(user_text)
    explanation = generate_explanation(theme)
    examples = generate_examples(theme)
    summary = generate_summary(theme)

    # すでに外部AIが長文を返している場合もあるので、導入文だけにしたいならここで調整してもよい
    intro = assistant_intro.strip()

    return f"{intro}\n{explanation}\n{examples}\n{summary}"


# =========================
# 入力ファイルのパース
# =========================
def load_dialogues(path: str) -> List[Tuple[str, str]]:
    """
    auto_generated_mix_train.txt のような形式から
    (user_text, assistant_text) のリストを作る。
    """
    dialogues: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        block_user = None
        block_assistant = None

        for line in f:
            line = line.rstrip("\n")

            if not line:
                # 空行で1ブロック終了
                if block_user is not None and block_assistant is not None:
                    dialogues.append((block_user, block_assistant))
                block_user = None
                block_assistant = None
                continue

            if line.startswith("User:"):
                block_user = line[len("User:"):].strip()
            elif line.startswith("Assistant:"):
                block_assistant = line[len("Assistant:"):].strip()
            else:
                # Assistant の複数行対応（箇条書きなど）
                if block_assistant is not None:
                    block_assistant += "\n" + line

        # ファイル末尾に空行がない場合のケア
        if block_user is not None and block_assistant is not None:
            dialogues.append((block_user, block_assistant))

    return dialogues


# =========================
# メイン処理
# =========================
def main(
    input_path: str = "auto_generated_mix_train.txt",
    output_path: str = "mix_train_completed.txt",
):
    dialogues = load_dialogues(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (user_text, assistant_text) in enumerate(dialogues, start=1):
            completed = complete_assistant_reply(user_text, assistant_text)

            f.write(f"User: {user_text}\n")
            f.write(f"Assistant: {completed}\n\n")

            print(f"[{i}/{len(dialogues)}] 補完完了")

    print(f"\n完了: {len(dialogues)} 件を補完 → {output_path}")


if __name__ == "__main__":
    main()

