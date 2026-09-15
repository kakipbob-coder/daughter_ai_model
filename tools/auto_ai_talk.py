import os
import random
import time
import requests
import google.generativeai as genai

# =========================
# Gemini API 設定
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-1.5-flash"

# =========================
# Ollama API 設定
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# =========================
# 安定版 Gemini 呼び出し
# =========================
def call_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)

        # text がある場合
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # fallback: parts から抽出
        if hasattr(response, "candidates"):
            parts = response.candidates[0].content.parts
            text = "".join([p.text for p in parts if hasattr(p, "text")])
            if text:
                return text.strip()

        return ""
    except Exception as e:
        print(f"[WARN] Gemini error: {e}")
        return ""


# =========================
# 安定版 Ollama 呼び出し
# =========================
def call_ollama(prompt: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=20
        )
        data = resp.json()
        text = data.get("response") or data.get("message") or ""
        return text.strip()
    except Exception as e:
        print(f"[WARN] Ollama error: {e}")
        return ""


# =========================
# フェイルオーバー + リトライ
# =========================
def call_external_ai(prompt: str, retries: int = 3) -> str:
    for _ in range(retries):
        # 1. Gemini
        text = call_gemini(prompt)
        if text:
            return text

        # 2. Ollama
        text = call_ollama(prompt)
        if text:
            return text

        time.sleep(1)

    return "（生成失敗）"


# =========================
# テーマ候補（後で増やせる）
# =========================
TOPICS = [
    "勉強の集中力を上げる方法",
    "睡眠の質を上げる方法",
    "猫の飼い方",
    "料理のコツ",
    "仕事のストレスとの付き合い方",
    "人間関係の悩み",
    "メンタルの整え方",
    "運動習慣を身につける方法",
    "時間管理のコツ",
    "仕事の効率を上げる方法",
]


# =========================
# User 発話生成
# =========================
def generate_user_utterance(topic: str) -> str:
    prompt = (
        f"あなたは相談者(User)です。テーマ「{topic}」について、"
        "自然な日本語で1〜2文の相談文を書いてください。"
        "必ずテーマに沿った内容にしてください。"
        "敬語で、丁寧な口調でお願いします。"
    )
    return call_external_ai(prompt)


# =========================
# Assistant 返答生成
# =========================
def generate_assistant_reply(user_text: str) -> str:
    prompt = (
        "あなたは相談に答えるアシスタント(Assistant)です。\n"
        "以下のUserの相談に対して、丁寧な日本語で3〜6文の回答を書いてください。\n"
        "必要に応じて箇条書きも使って構いません。\n"
        "内容は必ず User の相談内容に沿ったものにしてください。\n\n"
        f"User: {user_text}\n"
    )
    return call_external_ai(prompt)


# =========================
# メイン処理
# =========================
def main(output_path: str = "auto_generated_mix_train.txt", num_dialogues: int = 100):
    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(num_dialogues):
            topic = random.choice(TOPICS)

            user_text = generate_user_utterance(topic)
            assistant_text = generate_assistant_reply(user_text)

            f.write(f"User: {user_text}\n")
            f.write(f"Assistant: {assistant_text}\n\n")

            print(f"[{i+1}/{num_dialogues}] topic={topic}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()

