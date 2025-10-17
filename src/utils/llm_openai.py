import os
import backoff
from typing import List, Dict, Optional, Iterator, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None

@backoff.on_exception(backoff.expo, Exception, max_tries=2)
def stream_text(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 350,
) -> Tuple[Optional[Iterator[str]], Optional[str]]:
    client = _client()
    if client is None:
        return None, "No OPENAI_API_KEY set."

    try:
        stream_mgr = client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        def _iter() -> Iterator[str]:
            with stream_mgr as stream:
                for event in stream:
                    # ✅ New SDK (>=1.47)
                    if hasattr(event, "type") and event.type == "message.delta":
                        content = getattr(event.delta, "content", None)
                        if content:
                            yield content
                    # ✅ Older SDK fallback (<1.47)
                    elif hasattr(event, "choices"):
                        choice = event.choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta and getattr(delta, "content", None):
                            yield delta.content

        return _iter(), None

    except Exception as e:
        return None, str(e)

def complete_text(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 350,
) -> Tuple[Optional[str], Optional[str]]:
    client = _client()
    if client is None:
        return None, "No OPENAI_API_KEY set."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if hasattr(resp, "choices"):
            return (resp.choices[0].message.content or "").strip(), None
        elif hasattr(resp, "output") and hasattr(resp.output[0], "content"):
            return resp.output[0].content[0].text.strip(), None
        return None, "Unexpected response format."
    except Exception as e:
        return None, str(e)
