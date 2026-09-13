from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\-\+\.]+")

STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "a",
    "an",
    "with",
    "on",
    "is",
    "are",
    "by",
    "from",
    "at",
    "as",
}

# Terms used by local case reports.  Regexes that consume an entire Chinese
# sentence as one token make lexical matching almost always zero; these terms
# provide stable clinical boundaries without adding a heavyweight tokenizer.
CLINICAL_ZH_TERMS = (
    "非小细胞肺癌", "小细胞肺癌", "肺腺癌", "肺鳞癌", "乳腺癌", "结直肠癌",
    "脑膜转移", "脑转移", "骨转移", "肝转移", "淋巴结转移", "胸膜转移",
    "基因突变", "基因扩增", "基因融合", "耐药突变", "外显子缺失",
    "奥希替尼", "伏美替尼", "阿美替尼", "埃克替尼", "吉非替尼", "厄洛替尼",
    "卡马替尼", "赛沃替尼", "特泊替尼", "培美曲塞", "卡铂", "顺铂",
    "靶向治疗", "联合治疗", "单药治疗", "一线治疗", "二线治疗", "既往治疗",
    "部分缓解", "完全缓解", "疾病稳定", "疾病进展", "临床反应", "客观缓解率",
    "无进展生存", "总生存期", "症状改善", "影像学", "脑脊液", "转氨酶升高",
    "肝毒性", "不良反应", "停药", "减量",
)


def retrieval_terms(text: str, *, max_terms: int = 64) -> List[str]:
    """Return bilingual lexical terms suitable for query/chunk matching.

    Known clinical phrases are kept intact. Remaining Chinese text contributes
    short overlapping n-grams, which makes unseen drug/disease names matchable
    without treating the whole sentence as a single token.
    """
    normalized = normalize_text(text).lower()
    terms: List[str] = []
    seen = set()

    def add(term: str) -> None:
        key = term.strip().lower()
        if len(key) < 2 or key in seen or key in STOPWORDS or len(terms) >= max_terms:
            return
        seen.add(key)
        terms.append(key)

    for token in re.findall(r"[a-z][a-z0-9_.+-]*|\d+(?:\.\d+)?(?:\s*(?:mg|g|%|周|月|天))?", normalized):
        add(token)
    for term in CLINICAL_ZH_TERMS:
        if term in normalized:
            add(term)
    for span in re.findall(r"[\u4e00-\u9fff]+", normalized):
        # Bigrams preserve recall for previously unseen Chinese entities;
        # four-character windows add useful precision for clinical phrases.
        for width in (2, 4):
            if len(span) < width:
                continue
            for start in range(0, len(span) - width + 1):
                add(span[start:start + width])
                if len(terms) >= max_terms:
                    return terms
    return terms


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    text = normalize_text(text).lower()
    tokens = [token for token in TOKEN_PATTERN.findall(text) if token and token not in STOPWORDS]
    # Index and query use the same overlapping CJK units, avoiding arbitrary
    # four-character boundaries while retaining unseen entity recall.
    for span in re.findall(r"[\u4e00-\u9fff]+", text):
        for width in (2, 3, 4):
            if len(span) < width:
                continue
            tokens.extend(span[start:start + width] for start in range(len(span) - width + 1))
    return tokens


def softmax(scores: Iterable[float]) -> List[float]:
    values = list(scores)
    if not values:
        return []
    max_value = max(values)
    shifted = [math.exp(v - max_value) for v in values]
    total = sum(shifted)
    if total == 0:
        return [0.0 for _ in shifted]
    return [value / total for value in shifted]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    chunks: List[str] = []
    start = 0
    text_len = len(normalized)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(normalized[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def weighted_sum(score_map: Dict[str, float], weight_map: Dict[str, float]) -> float:
    total = 0.0
    for key, value in score_map.items():
        total += value * weight_map.get(key, 1.0)
    return total


def min_max_normalize(score_map: Dict[str, float]) -> Dict[str, float]:
    """Normalize one retrieval channel's scores to [0, 1]."""
    if not score_map:
        return {}

    minimum = min(score_map.values())
    maximum = max(score_map.values())
    if maximum == minimum:
        normalized_value = 1.0 if maximum > 0.0 else 0.0
        return {key: normalized_value for key in score_map}

    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in score_map.items()}
