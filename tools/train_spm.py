# -*- coding: utf-8 -*-
"""
SentencePiece tokenizer training script
"""

import sentencepiece as spm
import os

def train_spm():
    input_path = "data/train.txt"
    model_prefix = "tokenizer/tokenizer"
    vocab_size = 32000  # 俊光の100Mモデルに最適

    os.makedirs("tokenizer", exist_ok=True)

    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        num_threads=8,
        train_extremely_large_corpus=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )

    print("Tokenizer training complete.")
    print("Generated files:")
    print(" - tokenizer/tokenizer.model")
    print(" - tokenizer/tokenizer.vocab")

if __name__ == "__main__":
    train_spm()
