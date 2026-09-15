# run_pipeline.py

import subprocess
import sys
from pathlib import Path


RAW_PATH = Path("auto_generated_mix_train.txt")
COMPLETED_PATH = Path("mix_train_completed.txt")


def run_generate(num_dialogues: int = 100):
    """
    外部AIで会話ログを生成するフェーズ。
    generate_dialogues.py を呼び出して RAW_PATH を作る。
    """
    cmd = [sys.executable, "generate_dialogues.py"]
    print(f"[INFO] generate_dialogues.py 実行: {cmd}")
    subprocess.run(cmd, check=True)


def run_complete():
    """
    補完パイプラインフェーズ。
    complete_pipeline.py を呼び出して COMPLETED_PATH を作る。
    """
    cmd = [sys.executable, "complete_pipeline.py"]
    print(f"[INFO] complete_pipeline.py 実行: {cmd}")
    subprocess.run(cmd, check=True)


def main():
    print("[STEP 1] 外部AIで会話ログ生成")
    run_generate()

    if not RAW_PATH.exists():
        print(f"[ERROR] RAW ファイルが見つからない: {RAW_PATH}")
        return

    print("[STEP 2] 補完パイプラインで完成データ生成")
    run_complete()

    if COMPLETED_PATH.exists():
        print(f"[DONE] 完成データ: {COMPLETED_PATH}")
    else:
        print("[WARN] 完成データが生成されていないかも")


if __name__ == "__main__":
    main()

