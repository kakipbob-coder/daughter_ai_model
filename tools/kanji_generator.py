# -*- coding: utf-8 -*-
"""
JIS X 0208 の区点順で kanji.txt を生成する
"""

from jis0208_parser import parse_jis0208_unicode_txt

def generate_kanji_txt(jis_path, output_path):
    mapping = parse_jis0208_unicode_txt(jis_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for ku in range(1, 95):      # 1〜94区
            for ten in range(1, 95): # 1〜94点
                char = mapping.get((ku, ten))
                if char:
                    f.write(char)

    print(f"kanji.txt を生成しました → {output_path}")


if __name__ == "__main__":
    generate_kanji_txt("jis0208.txt", "kanji.txt")
