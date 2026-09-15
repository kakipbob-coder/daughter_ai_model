"""
前処理スクリプトの骨格。
必要に応じてクリーニングや正規化を追加する。
"""

def clean_text(text: str) -> str:
    # 必要なら後で実装
    return text


def preprocess_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = [clean_text(line.strip()) for line in lines if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")
