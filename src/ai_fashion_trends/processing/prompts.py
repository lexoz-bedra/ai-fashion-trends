"""Промпт-шаблоны для извлечения трендов из текстов."""

TREND_EXTRACTION_PROMPT = """\
Extract fashion trends from the text below. Return ONLY a JSON array.

Each object MUST have exactly these fields:
{{"category":"<one of: clothing|footwear|accessory|fabric|color|pattern|aesthetic|brand>","item":"<specific name>","sentiment":"<positive|negative|neutral>","confidence":<0.0-1.0>,"context_snippet":"<quote max 150 chars>","tags":["tag1","tag2"]}}

Example output:
[{{"category":"footwear","item":"red shoes","sentiment":"positive","confidence":0.9,"context_snippet":"Red shoes are trending this spring","tags":["spring","wedding"]}}]

Rules:
- ONLY fashion items (clothing, shoes, styles, aesthetics, colors, fabrics, brands)
- Skip: skincare, horoscopes, food, politics
- If no fashion trends found, return: []
- Return ONLY the JSON array, nothing else

Text:
---
{title}

{text}
---
"""


def build_prompt(title: str, text: str) -> str:
    """Сформировать промпт для извлечения трендов из одного поста."""
    truncated_text = text[:1500] if text else ""
    truncated_title = title[:200] if title else ""
    return TREND_EXTRACTION_PROMPT.format(title=truncated_title, text=truncated_text)
