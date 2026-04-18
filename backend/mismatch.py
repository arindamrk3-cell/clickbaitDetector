def check_semantic_mismatch(title, image_text):
    if not image_text.strip():
        return None  # no OCR text

    title_words = set(title.lower().split())
    image_words = set(image_text.lower().split())

    overlap = title_words.intersection(image_words)

    # If very little overlap → mismatch
    return len(overlap) < 2