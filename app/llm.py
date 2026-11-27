"""Thin wrapper around any OpenAI-compatible chat API."""
import json
import re

from openai import BadRequestError, OpenAI

from app.config import LLM_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY and not OPENAI_BASE_URL:
            raise RuntimeError(
                "No API key found. Copy .env.example to .env and set OPENAI_API_KEY "
                "(or OPENAI_BASE_URL for a local server)."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY or "not-needed", base_url=OPENAI_BASE_URL)
    return _client


def chat(messages: list[dict], json_mode: bool = False) -> str:
    kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def chat_json(system: str, user: str) -> dict:
    """Ask for a JSON object and parse it. Returns {} if the model's answer isn't usable."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        text = chat(messages, json_mode=True)
    except BadRequestError as error:
        # some servers don't support response_format, so try again without it
        if "response_format" not in str(error).lower():
            raise
        text = chat(messages)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}
