# -*- coding: utf-8 -*-
"""
JIS X 0208 → Unicode マッピングパーサ
（Unicode.org の JIS0208.TXT 専用）

使い方:
    mapping = parse_jis0208_unicode_txt("JIS0208.TXT")
    print(lookup(mapping, 55, 23))
    print(reverse_lookup(mapping, "魅"))
"""

def parse_jis0208_unicode_txt(path):
    """
    Unicode.org の JIS0208.TXT を読み込み、
    区点番号 → Unicode → 文字 の辞書を返す。
    """
    mapping = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # parts[1] = JISコード (例: 0x2121)
            # parts[2] = Unicode (例: 0x3000)
            jis_hex = parts[1]
            uni_hex = parts[2]

            # JISコードを整数化
            jis = int(jis_hex, 16)

            # 区点番号へ変換
            ku = (jis >> 8) - 0x20
            ten = (jis & 0xFF) - 0x20

            # Unicode → 文字
            uni = int(uni_hex, 16)
            char = chr(uni)

            mapping[(ku, ten)] = char

    return mapping


def lookup(mapping, ku, ten):
    """区点番号 → 文字"""
    return mapping.get((ku, ten))


def reverse_lookup(mapping, target_char):
    """文字 → 区点番号"""
    for (ku, ten), char in mapping.items():
        if char == target_char:
            return ku, ten
    return None


if __name__ == "__main__":
    # 動作テスト（必要ならコメント解除）
    # mapping = parse_jis0208_unicode_txt("JIS0208.TXT")
    # print("55区23点 =", lookup(mapping, 55, 23))
    # print("79区57点 =", lookup(mapping, 79, 57))
    # print("泪 =", reverse_lookup(mapping, "泪"))
    # print("魅 =", reverse_lookup(mapping, "魅"))
    pass
