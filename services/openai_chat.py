# services/openai_chat.py
import os
from openai import OpenAI

_client = None

def _client_once():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def chat_fast(prompt: str, user_id: str | None = None, max_tokens: int = 400) -> str:
    """
    Fast Q&A using gpt-4o-mini. Keep responses concise and safe for Discord.
    """
    client = _client_once()
    try:
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": (
                    "You are NextPlay Chat, a concise finance **education-only** chatbot for Discord. "
                    "Your role is to provide short, clear, and non-prescriptive answers. "
                    "Do not give allocations (percentages), buy/sell instructions, or product recommendations. "
                    "Instead, explain concepts in general terms using phrases like 'Some investors…' or 'Historically, people have…'. "
                    "Always append this disclaimer at the end of your answer: "
                    "'*This information is for educational purposes only and not financial advice. Please consult a licensed financial professional before making any investment decisions.*'"
                )},

                {"role": "user", "content": prompt},
            ],
            user=user_id if user_id else None,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        # Minimal surface; real details stay in logs
        return "Sorry, I hit an issue talking to the model. Please try again."
