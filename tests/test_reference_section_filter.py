from searchagent_retrieval.schema_parser import is_reference_list_chunk, strip_reference_section


def test_strips_trailing_english_references() -> None:
    text = "Body cites reference 1 in context.\n\nReferences\n1. Author. Paper. DOI"
    assert strip_reference_section(text) == "Body cites reference 1 in context."


def test_strips_numbered_chinese_reference_heading() -> None:
    text = "病例正文\n治疗后部分缓解。\n\n6. 参考文献\n[1] 某某等"
    assert strip_reference_section(text) == "病例正文\n治疗后部分缓解。"


def test_does_not_strip_inline_reference_word() -> None:
    text = "The reference cohort was used for comparison, and the patient responded."
    assert strip_reference_section(text) == text


def test_strips_references_and_notes_heading() -> None:
    text = "Article body.\n\nREFERENCES AND NOTES\n1. Author. [DOI]"
    assert strip_reference_section(text) == "Article body."


def test_strips_spaced_chinese_reference_heading() -> None:
    text = "病例正文\n\n参 考 文 献\n[1] 某某等"
    assert strip_reference_section(text) == "病例正文"


def test_detects_unheaded_reference_list_chunk() -> None:
    text = "A. Study. [DOI] [PubMed]\nB. Study. [DOI]"
    assert is_reference_list_chunk(text)


def test_does_not_treat_normal_body_as_reference_list() -> None:
    assert not is_reference_list_chunk("The body contains one supporting link [PubMed].")
