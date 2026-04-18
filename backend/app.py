from flask import Flask, request, jsonify
from flask_cors import CORS
from matplotlib.pyplot import text, title
import numpy as np,os
import pickle
#from mismatch import check_semantic_mismatch
from explain import get_explanation
from ocr import extract_text_from_image
from thumbnail_logic import detect_thumbnail_signal
from mismatch import check_semantic_mismatch
from audio_extractor import extract_audio
from speech_to_text import audio_to_text
from youtube_utils import get_video_info, download_audio, download_thumbnail, clean_youtube_url
from sentence_transformers import SentenceTransformer, util
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
app=Flask(__name__)
CORS(app)

model=pickle.load(open("model.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)

@app.route("/")
def home():
    return "Welcome to the Clickbait Detection API!"


@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    text=data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    vec=vectorizer.transform([text])
    prediction=model.predict(vec)[0]
    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = float(np.max(prob))
    else:
        confidence = None
    explanation=get_explanation(text)
    print(type(confidence), confidence)
    print("REASONS:", explanation)
    print("TYPES:", [type(r) for r in explanation])
    return jsonify({
        "prediction":"clickbait" if prediction==1 else "not clickbait",
        "confidence":float(confidence) if confidence is not None else 0.0,
        "explanation":[str(r) for r in explanation]
    })




@app.route("/predict_thumbnail",methods=["POST"])
def predict_thumbnail():
    try:
        text=request.form.get("text","")
        file=request.files.get("image")
        extracted_text=""
        if file:
            filepath=os.path.join(UPLOAD_FOLDER,file.filename)
            file.save(filepath)
            extracted_text=extract_text_from_image(filepath)
        
        if len(extracted_text.strip())>3:
            combined_text=text+ " "+extracted_text
            ocr_used=True
        else:
            combined_text=text
            ocr_used=False
        vec=vectorizer.transform([combined_text])
        pred=model.predict(vec)[0]
        confidence=0.0
        if hasattr(model,"predict_proba"):
            prob = model.predict_proba(vec)[0]
            confidence = float(np.max(prob))
        explanations=set(get_explanation(combined_text))
        if ocr_used:
            explanations.add("Text detected from thumbnail")
        else:
            explanations.add("Thumbnail text unclear")
        thumbnail_signals=detect_thumbnail_signal(extracted_text)
        explanations.update(thumbnail_signals)
        # ======================
        # 🎯 CLICKBAIT SCORE
        # ======================
        score = 0
        # Thumbnail signals
        if "Money-related claim detected" in thumbnail_signals:
            score += 2
        if "Unrealistic earnings claim" in thumbnail_signals:
            score += 2
        if "Free offer detected" in thumbnail_signals:
            score += 2
        # Text signals
        text_lower = combined_text.lower()
        if "secret" in text_lower:
            score += 1
        if "revealed" in text_lower:
            score += 1
        if "shocking" in text_lower:
            score += 1
        if "!" in combined_text:
            score += 1
        mismatch = check_semantic_mismatch(text, extracted_text)
        if mismatch is True:
            explanations.add("Title and thumbnail mismatch detected")
        elif mismatch is False:
            explanations.add("Title and thumbnail are aligned")


        strong_thumbnail_signal = False

        if ("Money-related claim detected" in thumbnail_signals or 
            "Free offer detected" in thumbnail_signals or 
           "Unrealistic earnings claim" in thumbnail_signals
        ):
            strong_thumbnail_signal = True
        clickbait_keywords = ["secret", "revealed", "shocking", "unbelievable"]
        for word in clickbait_keywords:
            if word in combined_text.lower():
                strong_thumbnail_signal = True
                break
        if strong_thumbnail_signal or mismatch is True:
            pred = 1   
            #explanations.add("Prediction adjusted due to strong thumbnail clickbait signals")
            if strong_thumbnail_signal:
                explanations.add("Prediction adjusted due to strong thumbnail clickbait signals")
            if mismatch is True:
                explanations.add("Prediction adjusted due to title-thumbnail mismatch")
        return jsonify({
            "prediction":"clickbait" if int(pred)==1 else "not clickbait",
            "confidence":confidence,
            "explanation":[str(r) for r in explanations],
            "clickbait_score": score
        })
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/predict_video", methods=["POST"])
def predict_video():
    print("Received request for /predict_video")
    try:
        text = request.form.get("text", "")
        file = request.files.get("video")
        print("Received text:", text)
        print("video received")
        if not file:
            return jsonify({"error": "No video uploaded"}), 400

        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)
        print("Video saved to:", video_path)
        audio_path = extract_audio(video_path)
        print("Audio extracted to:", audio_path)
        speech_text = audio_to_text(audio_path)
        print("Speech text extracted:", speech_text)
        combined_text = text + " " + speech_text
        print("Combined text:", combined_text)
        vec = vectorizer.transform([combined_text])
        pred = model.predict(vec)[0]
        print("Prediction:", pred)
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0]
            confidence = float(max(prob))
        explanations = set(get_explanation(combined_text))
        clickbait_keywords = ["secret", "revealed", "shocking", "money"]

        for word in clickbait_keywords:
            if word in combined_text.lower():
                pred = 1
                explanations.add("Prediction adjusted due to strong audio/text clickbait indicators")
                break

        return jsonify({
            "prediction": "clickbait" if int(pred) == 1 else "not clickbait",
            "confidence": confidence,
            "speech_text": speech_text,
            "explanation": list(explanations)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/predict_youtube", methods=["POST"])
def predict_youtube():
    try:
        data = request.get_json()
        url = data.get("url")
        url = clean_youtube_url(url)
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        title, thumbnail_url = get_video_info(url)
        thumb_path = download_thumbnail(thumbnail_url)
        audio_path = download_audio(url)
        speech_text = audio_to_text(audio_path)
        extracted_text = extract_text_from_image(thumb_path)
        if not speech_text.strip():
            combined_text = title
        else:
            combined_text = title + " " + speech_text
        if len(extracted_text.strip()) > 3:
            combined_text += " " + extracted_text
        combined_text = title + " " + extracted_text + " " + speech_text
        vec = vectorizer.transform([combined_text])
        pred = model.predict(vec)[0]
        explanations = set(get_explanation(combined_text))
        return jsonify({
            "prediction": "clickbait" if int(pred)==1 else "not clickbait",
            "title": title,
            "speech_text": speech_text,
            "thumbnail_text": extracted_text,
            "explanation": list(explanations)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def semantic_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0

    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    return score


def semantic_mismatch(title, other_text):
    if not other_text.strip():
        return None

    score = semantic_similarity(title, other_text)

    if score < 0.35:
        return True
    else:
        return False
@app.route("/predict_final", methods=["POST"])
def predict_final():
    try:
        text = request.form.get("text", "")
        image = request.files.get("image")
        video = request.files.get("video")

        extracted_text = ""
        speech_text = ""
        if image:
            img_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(img_path)

            extracted_text = extract_text_from_image(img_path)
        if video:
            video_path = os.path.join(UPLOAD_FOLDER, video.filename)
            video.save(video_path)

            audio_path = extract_audio(video_path)
            if audio_path:
                speech_text = audio_to_text(audio_path)
        combined_text = text

        if speech_text.strip():
            combined_text += " " + speech_text

        if len(extracted_text.strip()) > 3:
            combined_text += " " + extracted_text
        vec = vectorizer.transform([combined_text])
        pred = model.predict(vec)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0]
            confidence = float(max(prob))
        bollywood_keywords=[
            "bollywood","salman","shah rukh","srk","deepika","ranbir","alia","katrina"
        ]
        explanations = set(get_explanation(combined_text))
        if any(word in combined_text.lower() for word in bollywood_keywords):
            explanations.add("Bollywood releted content detected")
        # OCR info
        if extracted_text:
            explanations.add("Text detected from thumbnail")

        # Audio info
        if speech_text:
            explanations.add("Speech detected from video")
        thumbnail_signals = detect_thumbnail_signal(extracted_text)
        explanations.update(thumbnail_signals)
        #mismatch = check_semantic_mismatch(text, extracted_text)

        # if mismatch is True:
        #     explanations.add("Title and thumbnail mismatch detected")
        # elif mismatch is False:
        #     explanations.add("Title and thumbnail are aligned")
        mismatch_thumb = semantic_mismatch(text, extracted_text)
        mismatch_audio = semantic_mismatch(text, speech_text)
        
        if mismatch_thumb:
            explanations.add("Title and thumbnail semantic mismatch detected")
        
        if mismatch_audio:
            explanations.add("Title and audio semantic mismatch detected")
        strong_thumbnail_signal = False

        if (
            "Money-related claim detected" in thumbnail_signals or
            "Unrealistic earnings claim" in thumbnail_signals or
            "Free offer detected" in thumbnail_signals
        ):
            strong_thumbnail_signal = True

        # Keyword override
        keywords = ["secret", "revealed", "shocking", "money","viral","leaked","truth","exclusive","unseen","controversy","breaks internet"]

        for word in keywords:
            if word in combined_text.lower():
                strong_thumbnail_signal = True
                break

        if strong_thumbnail_signal or (mismatch_thumb is True) or (mismatch_audio is True):
            pred = 1
            explanations.add("Prediction adjusted using semantic and hybrid signals")
        score = 0

        if "Money-related claim detected" in thumbnail_signals:
            score += 2

        if "Unrealistic earnings claim" in thumbnail_signals:
            score += 2

        if "Free offer detected" in thumbnail_signals:
            score += 2

        text_lower = combined_text.lower()

        for word in ["secret", "revealed", "shocking"]:
            if word in text_lower:
                score += 1
        return jsonify({
            "prediction": "clickbait" if int(pred) == 1 else "not clickbait",
            "confidence": confidence,
            "clickbait_score": score,
            "title": text,
            "speech_text": speech_text,
            "thumbnail_text": extracted_text,
            "explanation": list(explanations)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__=="__main__":
    app.run(debug=True)