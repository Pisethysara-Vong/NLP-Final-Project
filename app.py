import re
from flask import Flask, render_template, request, jsonify
import PyPDF2
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from info_extraction import extract_skills, extract_job_org_relations, skills_pool


def preprocess_resume(text):
    # Remove emails and URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Replace numbers with placeholder
    text = re.sub(r'\d+', ' <NUM> ', text)
    
    # Keep letters, numbers, +, /, ., - 
    text = re.sub(r'[^a-zA-Z0-9\s\+\./-]', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize (regex tokenizer preserves C++, Node.js, etc.)
    tokenizer = RegexpTokenizer(r'\b\w[\w\+\./-]*\b')
    tokens = tokenizer.tokenize(text)
    
    # Stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ["would", "dont", "aaa"]
    stop_words.update(custom_stopwords)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
tfidf_vectorizer_MNB = pickle.load(open('./TF-IDF Models/tfidf_vectorizer_MNB.pkl', 'rb'))
tfidf_vectorizer_LR = pickle.load(open('./TF-IDF Models/tfidf_vectorizer_LR.pkl', 'rb'))
selector_MNB = pickle.load(open('./TF-IDF Models/selector_MNB.pkl', 'rb'))
selector_LR = pickle.load(open('./TF-IDF Models/selector_LR.pkl', 'rb'))
classifier_mnb = pickle.load(open('./TF-IDF Models/classifier_tfidf_mnb.pkl', 'rb'))
classifier_lr = pickle.load(open('./TF-IDF Models/classifier_tfidf_lr.pkl', 'rb'))


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    model_choice = request.form.get('model', 'mnb')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400
    
    # Extract text from PDF
    text = extract_text_from_pdf(file)
    if not text:
        return jsonify({'error': 'Could not extract text from PDF'}), 400
    
    # Preprocess
    processed_text = preprocess_resume(text)
    print("Finished preprocessing")
    
    # Vectorize
    # text_tfidf = tfidf_vectorizer.transform([processed_text])
    
    # Predict based on model choice
    if model_choice == 'mnb':
        text_tfidf = tfidf_vectorizer_MNB.transform([processed_text])
        text_fs_tfidf = selector_MNB.transform(text_tfidf)
        prediction = classifier_mnb.predict(text_fs_tfidf)[0]
        probabilities = classifier_mnb.predict_proba(text_fs_tfidf)[0]
        model_name = "Multinomial Naive Bayes"
    else:
        text_tfidf = tfidf_vectorizer_LR.transform([processed_text])
        text_fs_tfidf = selector_LR.transform(text_tfidf)
        prediction = classifier_lr.predict(text_fs_tfidf)[0]
        probabilities = classifier_lr.predict_proba(text_fs_tfidf)[0]
        model_name = "Logistic Regression"
    
    # Get top 3 predictions with probabilities
    classes = classifier_mnb.classes_ if model_choice == 'mnb' else classifier_lr.classes_
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_predictions = [
        {
            'category': classes[idx],
            'confidence': float(probabilities[idx] * 100)
        }
        for idx in top_3_indices
    ]
    
    # Extract skills and job-org relations
    skills = extract_skills(text, skills_pool)
    job_orgs = extract_job_org_relations(text)
    
    return jsonify({
        'model': model_name,
        'prediction': prediction,
        'confidence': float(max(probabilities) * 100),
        'top_predictions': top_predictions,
        'skills': skills,
        'job_org_relations': job_orgs,
        'text_preview': text[:500] + '...' if len(text) > 500 else text
    })

if __name__ == '__main__':
    app.run(debug=True)