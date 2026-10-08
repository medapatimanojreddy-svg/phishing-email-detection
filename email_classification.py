from flask import Flask, render_template, request, jsonify
import joblib
import socket
import re
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Set network timeout for socket lookups
socket.setdefaulttimeout(3.0)

# Base directory for resolving file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'model', 'phishing-model.joblib')
TFIDF_VECTORIZER_FILE = os.path.join(BASE_DIR, 'model', 'tfidf-vectorizer.joblib')

# Create Flask application
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load the trained phishing model and vectorizer
print("Loading phishing detection model...")
model = joblib.load(MODEL_FILE)
feature_extraction = joblib.load(TFIDF_VECTORIZER_FILE)
print("Model and vectorizer loaded successfully.")


def extract_urls(body):
    """Function to extract URLs from email body."""
    if not body:
        return []
    url_regex = r'(https?://[^\s<>"\')]+)'
    urls = re.findall(url_regex, str(body))
    # Remove trailing punctuation often captured with URLs
    cleaned = []
    for u in urls:
        u = u.rstrip('.,;:)')
        if u and u not in cleaned:
            cleaned.append(u)
    return cleaned


def resolve_url_to_ip(urls):
    """Function to resolve URLs to their IP addresses."""
    url_ip_dict = {}
    for url in urls:
        try:
            domain = url.split("//")[-1].split('/')[0].split('?')[0].split(':')[0]
            ip_address = socket.gethostbyname(domain)
            url_ip_dict[url] = ip_address
        except Exception:
            url_ip_dict[url] = "Unable to resolve"
    return url_ip_dict


def classify_text(subject, body):
    """Classify given subject and body as Phishing or Safe."""
    combined_text = f"{subject} {body}".strip()
    if not combined_text:
        return {
            'text': "No content provided to analyze.",
            'type': 'safe',
            'confidence': 100.0,
            'is_phishing': False,
            'urls': {}
        }

    input_data_features = feature_extraction.transform([combined_text])
    prediction = int(model.predict(input_data_features)[0])
    
    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_data_features)[0]
        confidence = round(float(proba[prediction]) * 100, 2)

    urls = extract_urls(body)
    url_ip_dict = resolve_url_to_ip(urls)

    is_phishing = (prediction == 1)
    status_label = "Phishing Email Detected!" if is_phishing else "Safe Email Received!"

    return {
        'subject': subject,
        'body_snippet': body[:200] + ('...' if len(body) > 200 else ''),
        'text': f"{status_label} Subject: {subject or 'No Subject'} (Confidence: {confidence}%)",
        'type': 'phishing' if is_phishing else 'safe',
        'is_phishing': is_phishing,
        'confidence': confidence,
        'urls': url_ip_dict
    }


@app.route('/', methods=['GET', 'POST'])
def home():
    single_result = None
    status_error = None
    input_subject = ""
    input_body = ""

    if request.method == 'POST':
        input_subject = request.form.get('subject', '').strip()
        input_body = request.form.get('body', '').strip()
        if input_subject or input_body:
            single_result = classify_text(input_subject, input_body)
        else:
            status_error = "Please provide an email subject or body to analyze."

    return render_template(
        'index.html',
        single_result=single_result,
        status_error=status_error,
        input_subject=input_subject,
        input_body=input_body
    )


@app.route('/api/classify', methods=['POST'])
def api_classify():
    data = request.get_json(force=True) or {}
    subject = data.get('subject', '')
    body = data.get('body', '')
    result = classify_text(subject, body)
    return jsonify(result)


if __name__ == '__main__':
    print("=" * 60)
    print("  PHISHING EMAIL DETECTION SYSTEM - ACTIVE")
    print("  Server URL: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
