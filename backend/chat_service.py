# -*- coding: utf-8 -*-
"""
chat_service.py
─────────────────
Ports DistilBot from app.py: calls Groq's OpenAI-compatible API if GROQ_API_KEY
is configured server-side, otherwise falls back to the same rule-based advisor
logic the original app used. The API key never leaves the backend / .env file.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

GROQ_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1"
MODEL_NAME_DEFAULT = "llama-3.3-70b-versatile"


def _get_api_cfg():
    key = os.getenv("GROQ_API_KEY", "")
    url = os.getenv("GROQ_BASE_URL", GROQ_BASE_URL_DEFAULT)
    model = os.getenv("MODEL_NAME", MODEL_NAME_DEFAULT)
    return key, url, model


def build_context_prompt(params: Dict, prediction: Optional[float] = None) -> str:
    pred_line = (f"Predicted ethanol purity: {prediction:.4f} ({prediction * 100:.2f} mol%)"
                 if prediction is not None else "No prediction run yet.")
    return f"""You are DistilBot, a helpful process-engineering assistant embedded inside an
ethanol distillation monitoring app. Speak in a friendly, expert but approachable tone.
Use short paragraphs. Never use bullet points unless asked. Avoid markdown headers.

Current column operating state:
  Pressure          : {params.get('pressure', 'N/A')} bar
  Top temperature   : {params.get('t1', 'N/A')} °C
  Reflux flow L     : {params.get('L', 'N/A')} kmol/hr
  Vapor flow V      : {params.get('V', 'N/A')} kmol/hr
  Distillate D      : {params.get('D', 'N/A')} kmol/hr
  Bottoms B         : {params.get('B', 'N/A')} kmol/hr
  Feed F            : {params.get('F', 'N/A')} kmol/hr
  Reflux ratio      : {params.get('reflux_ratio', 'N/A')}
  Reboiler intensity: {params.get('reboiler', 'N/A')}
  {pred_line}

Answer the user's question below based on this context."""


def _rule_based_response(msg: str, params: Dict, pred: Optional[float]) -> str:
    msg_l = msg.lower()
    rr = params.get("reflux_ratio", 0) or 0
    t1 = params.get("t1", 78)

    if pred is None:
        return ("Run a prediction first, then ask me about the result and I'll give you "
                "a detailed breakdown!")

    pct = pred * 100

    if any(w in msg_l for w in ["good", "bad", "result", "purity", "explain"]):
        quality = "excellent" if pred >= 0.85 else ("acceptable" if pred >= 0.75 else "below spec")
        tip = ""
        if pred < 0.75:
            tip = (f" Your reflux ratio is {rr:.2f} — try increasing it above 1.5. "
                   f"Also, top temperature of {t1}°C looks high; "
                   "reducing steam to the reboiler can help bring it down.")
        elif pred < 0.85:
            tip = (f" A small increase in reflux (currently {rr:.2f}) toward 1.8–2.0 "
                   "could push you into the optimal zone.")
        return (f"The predicted purity of {pct:.2f} mol% is {quality}.{tip} "
                "Want me to walk you through which parameter has the biggest lever?")

    if any(w in msg_l for w in ["improve", "increase", "higher"]):
        return (f"To raise purity from {pct:.1f}%, focus on three levers: "
                "1) raise the reflux ratio (increase L while keeping V steady), "
                "2) ensure top temperature stays near 78°C (ethanol boiling point at ~1 bar), "
                "3) avoid overloading the feed — keep F below 620 kmol/hr for best separation.")

    if any(w in msg_l for w in ["reflux", "ratio"]):
        return (f"Your reflux ratio is {rr:.3f}. A typical target for high-purity ethanol is 1.5–2.2. "
                f"{'This looks healthy!' if 1.5 <= rr <= 2.5 else 'Consider adjusting it into the 1.5–2.2 band.'}")

    if any(w in msg_l for w in ["temp", "temperature"]):
        return (f"Top tray temperature is {t1}°C. At ~1 bar, pure ethanol boils at 78.4°C. "
                f"{'You are right on target — great sign.' if 77 <= t1 <= 80 else 'Values above 82°C suggest water vapour in the distillate, which hurts purity.'}")

    return ("I'm DistilBot — ask me anything about your column! For example: "
            "'Is this purity good?', 'How do I improve it?', or 'What does reflux ratio mean?'")


def get_ai_response(user_message: str, params: Dict, prediction: Optional[float],
                     history: List[Dict[str, str]]) -> tuple[str, str]:
    """Returns (reply, source). source is 'groq', 'rule-based', or 'rule-based-fallback'."""
    api_key, base_url, model_name = _get_api_cfg()

    if not api_key:
        return _rule_based_response(user_message, params, prediction), "rule-based"

    try:
        from openai import OpenAI  # pip install openai
        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = [{"role": "system", "content": build_context_prompt(params, prediction)}]
        for h in history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=500, temperature=0.7
        )
        return resp.choices[0].message.content.strip(), "groq"

    except ImportError:
        return ("The `openai` package is not installed on the server. "
                "Run `pip install openai` then restart the backend.\n\n"
                + _rule_based_response(user_message, params, prediction)), "rule-based-fallback"
    except Exception as exc:  # noqa: BLE001
        return (f"AI API error: {exc}\n\n"
                + _rule_based_response(user_message, params, prediction)), "rule-based-fallback"
