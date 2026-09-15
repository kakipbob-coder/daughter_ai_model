# -*- coding: utf-8 -*-
"""
JIS X 0208 の完全データ（区点・JISコード・Unicode・文字）を
CSV / JSON にエクスポートするツール
"""

import json
from jis0208_parser import parse_jis0208_unicode_txt

def export_jis0208(jis_path, csv_path, json_path):
    mapping = parse_jis0208_unicode_txt(jis_path)

    rows = []
    for (ku, ten), char in sorted(mapping.items()):
        jis_code = (ku + 0x20) << 8 | (ten + 0x20)
        unicode_code = ord(char)

        rows.append({
            "ku": ku,
            "ten": ten,
            "jis_hex": f"0x{jis_code:04X}",
            "unicode_hex": f"U+{unicode_code:04X}",
            "char": char
        })

    # CSV 出力
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("ku,ten,jis_hex,unicode_hex,char\n")
        for r in rows:
            f.write(f"{r['ku']},{r['ten']},{r['jis_hex']},{r['unicode_hex']},{r['char']}\n")

    # JSON 出力
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"CSV 出力: {csv_path}")
    print(f"JSON 出力: {json_path}")


if __name__ == "__main__":
    export_jis0208("jis0208.txt", "jis0208_full.csv", "jis0208_full.json")
