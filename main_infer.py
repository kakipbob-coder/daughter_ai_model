from infer.generate import Generator
from utils.logging import get_logger


def main():
    logger = get_logger(__name__)

    model_path = "checkpoints/epoch_1.pt"  # 必要に応じて変更
    tokenizer_path = "tokenizer/tokenizer.model"

    gen = Generator(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )

    prompt = "こんにちは"
    out = gen.generate(prompt, max_new_tokens=50)

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Output: {out}")


if __name__ == "__main__":
    main()
