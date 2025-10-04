def redact_text(text: str) -> dict:
    """
    Input: raw text
    Output: dict:
      - clean_text (string)
      - redaction_map: list of {start, end, label, original_text}
    """
    raise NotImplementedError
