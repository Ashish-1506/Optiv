def extract_text_from_document(path: str) -> dict:
    """
    Input: path to .pptx, .xlsx, .pdf (digital)
    Output: dict with keys:
       - file_id
       - filename
       - mime
       - raw_text (string)
       - metadata (dict: pages, slides, etc.)
    """
    raise NotImplementedError
