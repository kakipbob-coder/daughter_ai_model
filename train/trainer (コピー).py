# File : /train/trainer.py
# Description :
#   Training pipeline for GPT-style language models.
#   Handles dataset loading, model initialization, optimizer/scheduler setup,
#   training loop execution, validation, checkpointing, and resume logic.
#
# 説明 :
#   GPT 系モデルの学習処理を統括するトレーナークラス。
#   データセットの読み込み、モデル構築、最適化関係の初期化、
#   学習ループの実行、検証（validation）、チェックポイント保存、
#   中断からの再開（resume）、進捗ログ出力など、
#   学習に必要な全処理を担当する。

import torch
from torch.utils.data import DataLoader
from src.dataset import TextDataset
from tokenizer.spm_wrapper import Tokenizer
from train.optimizer import build_optimizer
from train.scheduler import build_scheduler
from utils.checkpoint import save_checkpoint

from src.model import GPTModel
from datetime import datetime
import os
import time
import shutil  # ★ safe_save_checkpoint で使用


def log_with_time(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def safe_save_checkpoint(state, ckpt_path, logger=None):
    """
    atomic ではない FS も想定した、安全寄りの checkpoint 保存。
    tmp に保存 → 自分で load 検証 → OK なら本体に反映。
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    tmp_path = ckpt_path + ".tmp"
    backup_path = ckpt_path + ".prev"
    lock_path = os.path.join(ckpt_dir, ".ckpt_lock")

    # --- LOCK: 保存中フラグを立てる ---
    open(lock_path, "w").close()

    try:
        # 1. tmp に保存
        torch.save(state, tmp_path)

        # 2. tmp を自分で検証
        try:
            _ = torch.load(tmp_path, map_location="cpu")
        except Exception as e:
            if logger:
                logger.error(f"Checkpoint verify failed: {e}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return  # 本体は絶対に触らない

        # 3. 本体をバックアップ
        if os.path.exists(ckpt_path):
            try:
                shutil.copy2(ckpt_path, backup_path)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to backup checkpoint: {e}")

        # 4. 本体に反映
        shutil.copy2(tmp_path, ckpt_path)
        os.remove(tmp_path)

        if logger:
            logger.info(f"Safely saved checkpoint to {ckpt_path}")

    finally:
        # --- UNLOCK ---
        if os.path.exists(lock_path):
            os.remove(lock_path)

class Trainer:

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        count = 0

        total_batches = len(self.val_loader)

        for idx, (x, y) in enumerate(self.val_loader, start=1):
            x = x.to(self.device)
            y = y.to(self.device)

            logits, loss = self.model(x, y)
            total_loss += loss.item()
            count += 1

            # ★ validation 進捗ログ（100バッチごと）
            if idx % 100 == 0 or idx == total_batches:
                pct = (idx / total_batches) * 100
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                try:
                    with open("train_status.txt", "w") as f:
                        f.write(
                            f"val_progress={idx}/{total_batches} ({pct:.1f}%), "
                            f"time={now}"
                        )
                except Exception as e:
                    print(f"[WARN] failed to write train_status.txt: {e}")

                log_with_time(f"[val] progress {idx}/{total_batches} ({pct:.1f}%)")

        self.model.train()
        return total_loss / count


    def __init__(self, config):

        # checkpoint 保存間隔
        self.save_interval_step = 5000
        self.save_interval_time = 1800  # 30分

        # latest_step 用の時間保存間隔（秒）
        self.latest_save_interval = 600  # 10分
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = Tokenizer(config["tokenizer_path"])

        # Dataset
        self.train_dataset = TextDataset(
            config["train_data_path"],
            self.tokenizer,
            config["max_seq_len"],
            limit=None   # train は全データ使用（limit 機能は val 用）
        )
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        # ★ CHANGE: validation データセットを軽量化（limit があれば使う）
        val_limit = config.get("val_limit", None)  # 例: 50000
        self.val_dataset = TextDataset(
            config["val_data_path"],
            self.tokenizer,
            config["max_seq_len"],
            limit=val_limit  # ★ CHANGE
        )

        # ★ CHANGE: validation の batch_size を 2倍に
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["batch_size"] * 2,  # ★ CHANGE
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        # Model
        self.model = GPTModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"],
            max_seq_len=config["max_seq_len"],
        ).to(self.device)

        # Optimizer / Scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # resume 処理
        self.start_epoch = 1
        self.start_step = 1

        latest_path = "checkpoints/latest_step.pt"
        resume_path = latest_path if os.path.exists(latest_path) else config.get("resume_from", None)

        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device)

            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])

            self.start_epoch = ckpt.get("epoch", 1)
            self.start_step = ckpt.get("step", 1)

            # ★ latest_step / step checkpoint 共通の基準時刻
            self.last_ckpt_time = ckpt.get("time", time.time())
            self.last_latest_save = self.last_ckpt_time

            print(f"[INFO] Resume start epoch = {self.start_epoch}, step = {self.start_step}")
        else:
            # ★ 新規開始時の基準時刻
            self.last_ckpt_time = time.time()
            self.last_latest_save = self.last_ckpt_time

    def train(self):
        self.model.train()

        # last_ckpt_time / last_latest_save は __init__ で初期化済み
        # if not hasattr(self, "last_ckpt_time"):
        #     self.last_ckpt_time = time.time()

        for epoch in range(self.start_epoch, self.config["epochs"] + 1):

            for step, batch in enumerate(self.dataloader, start=1):

                if epoch == self.start_epoch and step < self.start_step:
                    continue

                x, y = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()

                logits, loss = self.model(x, y)

                if loss is None:
                    raise RuntimeError("Loss is None. Labels were not passed correctly.")

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

                self.optimizer.step()
                self.scheduler.step()

                if step % 10 == 0:
                    log_with_time(f"epoch {epoch} step {step} loss {loss.item():.4f}")

                if step % 50 == 0:
                    try:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open("train_status.txt", "w") as f:
                            f.write(f"epoch={epoch}, step={step}, time={now}")
                    except Exception as e:
                        print(f"[WARN] failed to write train_status.txt: {e}")

                # ★ CHANGE: 5000 stepごとの validation を削除
                # （ここは完全に削除してOK）

                # checkpoint 保存
                now_time = time.time()
                
                # 共通 state（Step 保存 / latest 保存 両方で使う）
                state = {
                    "epoch": epoch,
                    "step": step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "time": now_time,
                }
                
                # ★ Step 保存（5000 stepごと）: 履歴用
                if step % self.save_interval_step == 0:
                    lock_path = "checkpoints/.ckpt_lock"
                    open(lock_path, "w").close()

                    try:
                        save_path = f"checkpoints/step_checkpoint_e{epoch}_s{step}.pt"
                        torch.save(state, save_path)
                        print(f"[INFO] Saved step checkpoint: {save_path}")
                        self.last_ckpt_time = now_time
                    finally:
                        if os.path.exists(lock_path):
                            os.remove(lock_path)

                # ★ 時間保存（10分ごと）: resume 用 latest_step.pt
                if now_time - self.last_latest_save > self.latest_save_interval:
                    safe_save_checkpoint(state, "checkpoints/latest_step.pt")
                    print(f"[INFO] Updated latest_step.pt (time-based)")
                    self.last_latest_save = now_time

            # ★ CHANGE: エポック最後に1回だけ validation
            val_loss = self.evaluate()
            log_with_time(f"[val] epoch {epoch} val_loss={val_loss:.4f}")

            save_checkpoint(self.model, f"checkpoints/checkpoint_epoch{epoch}.pt")

            with open("epoch_done.flag", "w") as f:
                f.write(str(epoch))

        with open("training_done.flag", "w") as f:
            f.write("done")

