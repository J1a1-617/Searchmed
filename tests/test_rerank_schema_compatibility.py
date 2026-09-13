from searchagent_retrieval.llm_rerank import RERANK_SCHEMA


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_rerank_function_schema_avoids_gateway_unsupported_unique_items() -> None:
    assert all("uniqueItems" not in node for node in _walk(RERANK_SCHEMA))
