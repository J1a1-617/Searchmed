from __future__ import annotations

import re
from typing import List, Optional


_BOOLEAN_OPERATORS = frozenset({"and", "or", "not"})
_PUBMED_FIELDS = re.compile(
    r"\[(?:title(?:/abstract)?|abstract|tiab|mesh|mh|tw|dp|publication date)\]",
    flags=re.IGNORECASE,
)
_DATE_RANGE = re.compile(r"\b\d{4}\s*:\s*\d{4}\s*(?:\[dp\])?", flags=re.IGNORECASE)
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9_.+-]*|\d+(?:\.\d+)?(?:\s*(?:mg|g|%|weeks?|months?|days?))?|[\u4e00-\u9fff]{2,}",
    flags=re.IGNORECASE,
)


def _source_without_query_syntax(query: str) -> str:
    lines = []
    for line in str(query or "").splitlines():
        # Replanner occasionally emitted explanatory comments after a valid
        # query. They are not searchable clinical information.
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    text = " ".join(lines)
    text = _PUBMED_FIELDS.sub(" ", text)
    text = _DATE_RANGE.sub(" ", text)
    return text


def _content_tokens(query: str, *, max_tokens: Optional[int]) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for match in _TOKEN_PATTERN.findall(_source_without_query_syntax(query)):
        token = " ".join(str(match).split())
        key = token.lower()
        if not token or key in _BOOLEAN_OPERATORS or key in seen:
            continue
        seen.add(key)
        tokens.append(token)
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens


def adapt_dense_query(query: str, max_tokens: Optional[int] = None) -> str:
    """Remove unsupported syntax without truncating semantic query content."""
    tokens = _content_tokens(query, max_tokens=max_tokens)
    return " ".join(tokens) or " ".join(str(query or "").split())


def adapt_bm25_query(query: str, max_tokens: int = 64) -> str:
    """Keep exact clinical terms while removing unsupported Boolean syntax."""
    tokens = _content_tokens(query, max_tokens=max_tokens)
    return " ".join(tokens) or " ".join(str(query or "").split())[:800]
