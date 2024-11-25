from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import spacy
import re
from collections import Counter

app = Flask(__name__)

# Load models and NLP tools
MODEL_PATH = "saved_models/legal/best_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def extract_key_points(text):
    """Extract key legal points from the text."""
    doc = nlp(text)
    
    # Extract sentences with legal keywords
    legal_keywords = {
        'jurisdiction', 'provision', 'section', 'act', 'order',
        'appeal', 'review', 'authority', 'court', 'power'
    }
    
    key_sentences = []
    for sent in doc.sents:
        words = set(token.text.lower() for token in sent)
        if any(keyword in words for keyword in legal_keywords):
            key_sentences.append(sent.text)
    
    return key_sentences

def analyze_sections_cited(text):
    """Extract and analyze sections cited in the text."""
    section_pattern = r'(?:Section|S\.)\s*(\d+(?:\(\d+\))?(?:\s*of\s*the\s*[A-Za-z\s]+)?)'
    sections = re.findall(section_pattern, text)
    return Counter(sections)

def generate_explanation(text, prediction, confidence):
    """Generate a detailed explanation for the prediction."""
    key_points = extract_key_points(text)
    sections = analyze_sections_cited(text)
    
    explanation = {
        "decision": "Favorable" if prediction == 1 else "Unfavorable",
        "confidence": f"{confidence:.2%}",
        "key_points": key_points[:3],  # Top 3 key points
        "sections_cited": dict(sections.most_common(3)),  # Top 3 most cited sections
        "reasoning": []
    }
    
    # Add reasoning based on key legal indicators
    if "jurisdiction" in text.lower():
        explanation["reasoning"].append("Jurisdictional aspects were considered")
    if "final" in text.lower():
        explanation["reasoning"].append("Finality of orders was discussed")
    if "power" in text.lower():
        explanation["reasoning"].append("Authority/power considerations were present")
    
    return explanation

def predict_text(text):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    # Generate detailed explanation
    explanation = generate_explanation(text, predicted_class, confidence)
    
    return {
        "prediction": predicted_class,
        "prediction_text": explanation["decision"],
        "confidence": confidence,
        "explanation": explanation
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        result = predict_text(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        texts = data['texts']
        
        results = []
        for text in texts:
            prediction = predict_text(text)
            results.append({
                "text": text,
                "prediction": prediction["prediction"],
                "prediction_text": prediction["prediction_text"],
                "confidence": prediction["confidence"],
                "explanation": prediction["explanation"]
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)