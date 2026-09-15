# File: /infer/cli.py
# Description:
#   Command-line interface for running text generation using the Generator class.
#   Accepts model path, tokenizer path, prompt text, and decoding parameters.
#
# 説明:
#   Generator クラスを用いて推論を実行する CLI インターフェース。
#   モデルパス・トークナイザパス・プロンプト・生成設定を受け取り、
#   コマンドラインからテキスト生成を行う。
#
# Notes:
#   - generate.py の Generator を直接呼び出す実行エントリーポイント。
#   - top-k / temperature を CLI から調整可能。

import argparse
from infer.generate import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    gen = Generator(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
    )

    out = gen.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(out)


if __name__ == "__main__":
    main()
