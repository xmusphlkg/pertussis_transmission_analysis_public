from manuscript_notes.audit_submission_package import words


def test_word_count_excludes_html_superscript_citations() -> None:
    assert words("Evidence.<sup>1–3</sup> More evidence.<sup>4,5</sup>") == 3


def test_word_count_excludes_markdown_superscript_citations() -> None:
    assert words("Evidence.^1–3^ More evidence.^4,5^") == 3


def test_word_count_preserves_younger_than_comparison_text() -> None:
    text = "Effects on all-<18 cases.<sup>1</sup> Results remained stable."
    assert words(text) == 8


def test_word_count_handles_images_links_and_urls() -> None:
    text = (
        "See ![Figure caption](figure.png) the [study protocol](https://example.org) "
        "at https://example.org/source"
    )
    assert words(text) == 5
