# -*- coding: utf-8 -*-
"""
Shift_JIS / EUC-JP の変換表を生成する
"""

def parse_sjis_and_jis(path):
    """
    JIS0208.TXT を読み込み、
    SJIS → Unicode
    JIS → Unicode
    の2種類の辞書を返す
    """
    sjis_map = {}
    jis_map = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            sjis_hex = parts[0]   # 0x8140
            jis_hex  = parts[1]   # 0x2121
            uni_hex  = parts[2]   # 0x3000

            sjis = int(sjis_hex, 16)
            jis  = int(jis_hex, 16)
            uni  = int(uni_hex, 16)

            sjis_map[sjis] = uni
            jis_map[jis] = uni

    return sjis_map, jis_map


def jis_to_eucjp(jis_code):
    """
    JISコード (0x2121 など) → EUC-JP のバイト列を計算
    EUC-JP = JISコードの各バイトに 0x80 を足す
    """
    high = (jis_code >> 8) & 0xFF
    low  = jis_code & 0xFF
    return bytes([high + 0x80, low + 0x80])


def export_encodings(jis_path, sjis_csv, euc_csv):
    sjis_map, jis_map = parse_sjis_and_jis(jis_path)

    # Shift_JIS → Unicode
    with open(sjis_csv, "w", encoding="utf-8") as f:
        f.write("sjis_hex,unicode_hex,char\n")
        for sjis, uni in sorted(sjis_map.items()):
            f.write(f"0x{sjis:04X},U+{uni:04X},{chr(uni)}\n")

    # EUC-JP → Unicode
    with open(euc_csv, "w", encoding="utf-8") as f:
        f.write("eucjp_hex,unicode_hex,char\n")
        for jis, uni in sorted(jis_map.items()):
            euc = jis_to_eucjp(jis)
            euc_hex = "".join(f"{b:02X}" for b in euc)
            f.write(f"{euc_hex},U+{uni:04X},{chr(uni)}\n")

    print(f"Shift_JIS → Unicode: {sjis_csv}")
    print(f"EUC-JP → Unicode: {euc_csv}")


if __name__ == "__main__":
    export_encodings("jis0208.txt", "sjis_map.csv", "eucjp_map.csv")
