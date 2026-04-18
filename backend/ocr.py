import easyocr

# Initialize once (IMPORTANT)
reader = easyocr.Reader(['en'])

import re
def normalize_ocr_text(text):
    text = text.lower()

    # Fix common OCR mistakes
    text = text.replace("s100", "$100")
    text = text.replace("pfr", "per")
    text = text.replace("0", "0")

    # Remove noise
    text = re.sub(r'[^a-zA-Z0-9\s$]', '', text)

    return text.strip()
def extract_text_from_image(image_path):
    try:
        result = reader.readtext(image_path)

        # Extract only text
        texts = [res[1] for res in result]

        final_text = " ".join(texts)
        cleaned= normalize_ocr_text(final_text)
        return cleaned

    except Exception as e:
        print("EasyOCR Error:", e)
        return ""
    
