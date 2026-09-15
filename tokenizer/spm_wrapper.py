import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # ★ 追加：pad_id を取得
        self.pad_id = self.sp.pad_id()

        # ★ vocab_size をプロパティではなく属性として持たせる
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        # token_id のリストを返す
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        # token_id のリストからテキストへ
        return self.sp.decode(ids)
