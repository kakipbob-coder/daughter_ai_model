# File: /infer/chat_loop.py
# Description:
#   Simple interactive chat loop using GPTModel-based Generator.
#
# 説明:
#   GPTModel を用いた簡易対話ループ。
#   ユーザ入力とモデル出力を履歴として蓄積し、
#   ChatGPT 風の対話を実現する。

import argparse
from infer.generate import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    gen = Generator(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
    )

    print("=== ChatGPT-like interactive mode ===")
    print("Type 'exit' to quit.\n")

    # 小規模モデル向けに履歴を使わない方式に変更
    # history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # 履歴ではなく、毎ターン最新のペアだけを渡す
        prompt = f"User: {user_input}\nAssistant: "

        # モデル生成
        response = gen.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # モデル出力から「Assistant: 」以降を抽出
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
