MODEL_CONFIG = {
    "vocab_size": 24000,      # SentencePiece で作る語彙数
    "d_model": 512,           # 埋め込み次元
    "n_layer": 12,            # Transformer ブロック数
    "n_head": 8,              # Attention ヘッド数
    "d_ff": 2048,             # FFN の中間次元
    "max_seq_len": 1024,      # コンテキスト長
    "dropout": 0.1,           # Dropout（学習時のみ）
#    "rope_theta": 10000,      # RoPE の基本角度
}
