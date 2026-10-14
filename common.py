import time
from typing import Literal, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel

model_name="openai/gpt-oss-120b"
CALL_PAUSE = 13
RETRIES = 6


def get_llm(max_tokens=4096):
    return ChatGroq(model=model_name, max_tokens=max_tokens, temperature=0.3)


def with_retry(fn):
    for attempt in range(RETRIES):
        try:
            out = fn()
            time.sleep(CALL_PAUSE)
            return out
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"[model call failed: {str(e)[:150]}] retrying in {wait}s ({attempt + 1}/{RETRIES})")
            time.sleep(wait)
    raise RuntimeError("Model call failed after several retries")


def dialogue_to_str(messages: List[BaseMessage]) -> str:
    return "\n\n".join(f"{getattr(m, 'type', 'msg').upper()}: {m.content}" for m in messages)


def text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content)


def ask(system: str, human: str) -> str:
    out = with_retry(lambda: get_llm().invoke([SystemMessage(content=system), HumanMessage(content=human)]))
    return text_of(out)


class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str
    reason: str


def decide(system: str, human: str) -> Decision:
    llm = get_llm().with_structured_output(Decision)
    return with_retry(lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]))