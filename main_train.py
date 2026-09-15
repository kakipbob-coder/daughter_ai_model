# File : main_train.py
# Description :
#   Entry point for launching the training process.
#   Loads training configuration, initializes logging, sets random seed,
#   constructs the Trainer instance, and starts the training loop.
#
# 説明 :
#   学習処理のエントリーポイント。
#   ・TRAIN_CONFIG の読み込み
#   ・ログ出力（logger 初期化）
#   ・乱数シードの固定（再現性確保）
#   ・Trainer の生成
#   ・trainer.train() による学習開始
#
#   学習ロジック本体は trainer.py に集約されており、
#   このファイルは「学習開始のための最小限の制御」を担当する。

from train.trainer import Trainer
from config.train_config import TRAIN_CONFIG
from utils.seed import set_seed
from utils.logging import get_logger


def main():
    logger = get_logger(__name__)
    logger.info("Starting training...")

    # ★ 再現性確保のためのシード固定
    set_seed(TRAIN_CONFIG["seed"])

    # ★ Trainer を構築し、学習を開始
    trainer = Trainer(TRAIN_CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()

