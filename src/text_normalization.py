from __future__ import annotations

import re


TERM_TRANSLATIONS = {
    "tailings": "хвосты",
    "flotation": "флотация",
    "regrinding": "доизмельчение",
    "classification": "классификация",
    "desliming": "обесшламливание",
    "reagent regime": "реагентный режим",
    "recovery": "извлечение",
    "losses": "потери",
    "selectivity": "селективность",
    "particle size": "крупность",
    "copper": "медь Cu",
    "nickel": "никель Ni",
    "pH": "pH",
    "尾矿": "хвосты",
    "浮选": "флотация",
    "再磨": "доизмельчение",
    "分级": "классификация",
    "脱泥": "обесшламливание",
    "回收率": "извлечение",
    "损失": "потери",
    "选择性": "селективность",
    "粒度": "крупность",
    "铜": "медь Cu",
    "镍": "никель Ni",
}


def append_normalized_terms(text: str) -> str:
    additions = []
    for source, target in TERM_TRANSLATIONS.items():
        if _contains_term(text, source):
            additions.append(target)
    if not additions:
        return text
    unique = " ".join(dict.fromkeys(additions))
    return f"{text}\n\nНормализованные мультиязычные термины: {unique}"


def _contains_term(text: str, term: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", term):
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None
