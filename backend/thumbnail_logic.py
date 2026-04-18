def detect_thumbnail_signal(image_text):
    image_text = image_text.lower()

    signals = []

    if "$" in image_text or "100" in image_text:
        signals.append("Money-related claim detected")

    if "per day" in image_text:
        signals.append("Unrealistic earnings claim")

    if "free" in image_text:
        signals.append("Free offer detected")

    return signals