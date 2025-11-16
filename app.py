import re
from flask import Flask, render_template, request, jsonify
import PyPDF2
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# Comprehensive known companies whitelist
KNOWN_COMPANIES = {
    "amazon", "microsoft", "google", "facebook", "meta", "apple", "netflix", "tesla",
    "walmart", "target", "costco", "starbucks", "mcdonald's", "mcdonalds", "disney", 
    "coca-cola", "pepsi", "pepsico", "ibm", "oracle", "adobe", "salesforce", "uber", 
    "airbnb", "linkedin", "twitter", "ebay", "paypal", "visa", "mastercard", "amex",
    "boeing", "ge", "general electric", "ford", "toyota", "honda", "nike", "adidas",
    "mgm", "hilton", "marriott", "hyatt", "intercontinental",
    "henkel", "unilever", "procter & gamble", "p&g", "johnson & johnson",
    "pfizer", "merck", "abbvie", "bristol myers", "eli lilly",
    "cnn", "nbc", "abc", "cbs", "fox", "espn", "hbo", "msnbc", "techtv",
    "dentsu", "aegis", "omnicom", "wpp", "publicis", "havas",
    "caesars", "elizabeth arden", "remington", "allergan",
    "gaylord", "lesueur", "michael foods", "wavefly", "jmf solutions",
    "sysco", "aramark", "sodexo", "compass group", "jpmorgan", "wells fargo",
    "bank of america", "citigroup", "goldman sachs", "morgan stanley",
    "accenture", "deloitte", "pwc", "kpmg", "ey", "mckinsey", "bain", "bcg"
}

# Definite NOT companies (software, cities, programs, etc.)
NOT_COMPANIES = {
    # Software/Tools
    "kronos", "ulti pro", "ultipro", "workday", "adp", "sap", "oracle", "salesforce crm",
    "microsoft office", "ms office", "excel", "powerpoint", "word", "outlook",
    "photoshop", "illustrator", "indesign", "adobe suite",
    # Government programs
    "medicare", "medicaid", "social security", "osha", "ada", "fmla", "cobra",
    # Cities (common ones in resumes)
    "chaska", "minneapolis", "chicago", "seattle", "new york", "san francisco",
    "boston", "atlanta", "dallas", "houston", "phoenix", "denver",
    # Generic terms
    "state", "city", "county", "department", "agency", "commission",
    # Education
    "university", "college", "school", "institute"
}

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

def is_definitely_not_company(text):
    """Check if text is definitely NOT a company."""
    text_lower = text.lower().strip()
    
    # Check NOT_COMPANIES list
    if text_lower in NOT_COMPANIES:
        return True
    
    # Check if it contains software/tool names
    software_indicators = ["office", "excel", "word", "powerpoint", "outlook", "kronos", "ulti"]
    if any(sw in text_lower for sw in software_indicators):
        return True
    
    # Check if it contains education indicators
    edu_indicators = ["university", "college", "school", "institute", "academy"]
    if any(edu in text_lower for edu in edu_indicators):
        return True
    
    # Check if it contains location indicators
    location_indicators = ["city", "state", "county", "street", "avenue", "road"]
    if any(loc in text_lower for loc in location_indicators):
        return True
    
    return False

def is_likely_company(text):
    """Check if text is likely a company name."""
    text_lower = text.lower().strip()
    
    # Whitelist check
    if text_lower in KNOWN_COMPANIES:
        return True
    
    # Definitely not a company
    if is_definitely_not_company(text):
        return False
    
    # Must be reasonable length
    if len(text) < 2 or len(text) > 50:
        return False
    
    # Single character or very short
    if len(text) <= 2:
        return False
    
    # All caps short acronyms (unless whitelisted)
    if text.isupper() and len(text) <= 3:
        return False
    
    # Must start with capital or number
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    
    # If it's 2+ capitalized words, probably a company
    words = text.split()
    if len(words) >= 2:
        cap_words = [w for w in words if w[0].isupper()]
        if len(cap_words) >= 2:
            return True
    
    # Single word that's capitalized and reasonable length
    if len(words) == 1 and len(text) >= 4:
        return True
    
    return False

def extract_job_org_relations(text):
    """
    Multi-strategy extraction that handles various resume formats.
    """
    relations = []
    seen_companies = set()
    
    # Strategy 1: Explicit patterns with job titles
    # Patterns like "Senior HR Manager at Company Name" or "worked as X at Y"
    patterns = [
        r"(?:worked|served|employed|hired)\s+(?:as\s+)?([^,.\n]{5,50}?)\s+(?:at|for|with)\s+([A-Z][a-zA-Z\s&.''-]+?)(?:\s|,|\.|$)",
        r"([A-Z][a-zA-Z\s]+(?:manager|director|analyst|engineer|developer|designer|specialist|coordinator|consultant|representative|officer|supervisor|lead|architect))\s+(?:at|for|with)\s+([A-Z][a-zA-Z\s&.''-]+?)(?:\s|,|\.|$)",
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            job_title = match.group(1).strip()
            company = match.group(2).strip()
            
            # Clean company name
            company = re.split(r'\s+(?:in|from|since|during|-)', company)[0].strip()
            
            if is_likely_company(company) and company.lower() not in seen_companies:
                relations.append(f"{job_title.title()} at {company.title()}")
                seen_companies.add(company.lower())
    
    # Strategy 2: Look for whitelisted companies anywhere in text
    for known_company in KNOWN_COMPANIES:
        if known_company.lower() in seen_companies:
            continue
        
        # Case-insensitive search
        pattern = r'\b' + re.escape(known_company) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            # Check if it's in a work context (within 150 chars of work keywords)
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 150)
                context = text[start:end].lower()
                
                work_keywords = [
                    'work', 'employ', 'position', 'role', 'experience', 
                    'company', 'organization', 'corporation', 'firm',
                    'managed', 'led', 'developed', 'responsible', 'duties'
                ]
                
                if any(kw in context for kw in work_keywords):
                    relations.append(f"Experience at {known_company.title()}")
                    seen_companies.add(known_company.lower())
                    break
    
    # Strategy 3: Use spaCy but with very strict filtering
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        
        if ent.text.lower() in seen_companies:
            continue
        
        if not is_likely_company(ent.text):
            continue
        
        # Check work context in the sentence
        sent = ent.sent.text.lower()
        work_indicators = [
            'work', 'employ', 'position', 'role', 'experience', 
            'responsibilities', 'duties', 'managed', 'led', 'developed'
        ]
        
        if any(indicator in sent for indicator in work_indicators):
            relations.append(f"Experience at {ent.text.title()}")
            seen_companies.add(ent.text.lower())
    
    # Strategy 4: Last resort - look for capitalized 2-3 word phrases in work sections
    # This helps when the resume has company names but spaCy missed them
    work_section_pattern = r'(?:experience|employment|work history|professional background)[:\s]+([^\.]+?)(?:education|skills|certifications|$)'
    work_sections = re.findall(work_section_pattern, text, re.IGNORECASE | re.DOTALL)
    
    for section in work_sections:
        # Find capitalized multi-word phrases
        cap_phrases = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', section)
        
        for phrase in cap_phrases:
            if phrase.lower() in seen_companies:
                continue
            
            if is_likely_company(phrase):
                relations.append(f"Experience at {phrase.title()}")
                seen_companies.add(phrase.lower())
    
    return relations if relations else ["No clear job-org relationships found"]

# Skills extraction
skills_pool = {
    "HR": ["recruitment", "interviewing", "onboarding", "employee relations", "hr policies",
           "payroll", "talent acquisition", "performance management", "training", "benefits administration"],
    "DESIGNER": ["photoshop", "illustrator", "figma", "adobe xd", "ui design", "ux design", "wireframing",
                 "logo design", "graphic design", "typography", "prototyping", "creativity"],
    "INFORMATION-TECHNOLOGY": ["python", "java", "c++", "c#", "javascript", "react", "node.js", "sql", "html", "css",
                                "typescript", "flask", "django", "spring", "software development", "data analysis",
                                "machine learning", "ai", "pandas", "numpy", "git"],
    "TEACHER": ["lesson planning", "curriculum development", "classroom management", "grading",
                "student engagement", "communication", "mentoring", "public speaking", "education technology"],
    "ADVOCATE": ["legal research", "drafting", "litigation", "contracts", "compliance", "corporate law",
                 "intellectual property", "negotiation", "case management", "advocacy"],
    "BUSINESS-DEVELOPMENT": ["sales strategy", "client relations", "market research", "partnerships", "lead generation",
                             "b2b", "crm", "pipeline management", "business strategy", "negotiation"],
    "HEALTHCARE": ["patient care", "diagnosis", "medical records", "phlebotomy", "emergency response",
                   "clinical procedures", "first aid", "surgery assistance", "public health", "pharmacology"],
    "FITNESS": ["personal training", "nutrition", "fitness assessment", "strength training", "cardio",
                "injury prevention", "yoga", "exercise programming", "motivation", "client coaching"],
    "AGRICULTURE": ["crop management", "soil science", "irrigation", "harvesting", "farm equipment",
                    "organic farming", "agribusiness", "animal husbandry", "pesticide control", "planting"],
    "BPO": ["customer service", "inbound calls", "outbound calls", "ticketing", "crm software",
            "technical support", "communication", "time management", "problem solving", "escalation handling"],
    "SALES": ["sales strategy", "lead generation", "customer relationship management", "negotiation",
              "closing deals", "cold calling", "presentation", "sales forecasting", "upselling", "crm"],
    "CONSULTANT": ["business analysis", "project management", "strategy", "stakeholder management",
                   "data analysis", "presentation", "problem solving", "process improvement"],
    "DIGITAL-MEDIA": ["social media", "seo", "content creation", "copywriting", "google ads", "facebook ads",
                      "email marketing", "campaign management", "analytics", "branding"],
    "AUTOMOBILE": ["vehicle maintenance", "mechanical repair", "diagnostics", "automotive systems",
                   "engine tuning", "electrical systems", "safety inspection", "parts replacement"],
    "CHEF": ["cooking", "menu planning", "food safety", "recipe development", "plating", "baking",
             "inventory management", "teamwork", "sanitation", "knife skills"],
    "FINANCE": ["accounting", "budgeting", "financial analysis", "investment", "forecasting", "auditing",
                "taxation", "risk management", "bookkeeping", "excel", "financial modeling"],
    "APPAREL": ["fashion design", "textile", "sewing", "pattern making", "trend analysis", "styling",
                "merchandising", "fabric selection", "retail management"],
    "ENGINEERING": ["autocad", "solidworks", "matlab", "design analysis", "project management", "mechanical systems",
                    "civil design", "electrical circuits", "simulation", "blueprint reading"],
    "ACCOUNTANT": ["bookkeeping", "financial reporting", "tax preparation", "auditing", "reconciliation",
                   "payroll", "excel", "budgeting", "financial statements", "accounting software"],
    "CONSTRUCTION": ["site supervision", "blueprint reading", "safety management", "quantity surveying",
                     "project scheduling", "civil works", "materials management", "contract management"],
    "PUBLIC-RELATIONS": ["media relations", "press release", "branding", "crisis communication", "copywriting",
                         "public speaking", "event coordination", "marketing communication"],
    "BANKING": ["customer service", "loan processing", "credit analysis", "cash handling",
                "financial advisory", "branch operations", "risk assessment", "compliance"],
    "ARTS": ["painting", "drawing", "illustration", "creative writing", "music composition",
             "photography", "editing", "storytelling", "animation", "art direction"],
    "AVIATION": ["flight operations", "navigation", "safety procedures", "aircraft maintenance",
                 "air traffic communication", "crew management", "aviation regulations"]
}

def extract_skills(text, skills_pool):
    """Extract skills from text."""
    text_lower = text.lower()
    extracted_skills = []
    
    for skill in {s.lower() for v in skills_pool.values() for s in v}:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            extracted_skills.append(skill)
    
    skills_by_domain = {domain: [] for domain in skills_pool}
    for skill in extracted_skills:
        for domain, skill_list in skills_pool.items():
            if skill.lower() in [s.lower() for s in skill_list]:
                skills_by_domain[domain].append(skill.capitalize())
                break
    
    return skills_by_domain



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
classifier_mnb = pickle.load(open('classifier_tfidf_mnb.pkl', 'rb'))
classifier_lr = pickle.load(open('classifier_tfidf_lr.pkl', 'rb'))


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
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    
    # Predict based on model choice
    if model_choice == 'mnb':
        prediction = classifier_mnb.predict(text_tfidf)[0]
        probabilities = classifier_mnb.predict_proba(text_tfidf)[0]
        model_name = "Multinomial Naive Bayes"
    else:
        prediction = classifier_lr.predict(text_tfidf)[0]
        probabilities = classifier_lr.predict_proba(text_tfidf)[0]
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