from __future__ import annotations
import re

def clean_text(text: str) -> str:
    """Minimal cleaner; keep it simple and explainable."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
