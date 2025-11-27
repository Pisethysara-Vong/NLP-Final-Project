import re
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

# Tools and technologies that are NOT companies
TECH_TOOLS = {
    "python", "java", "javascript", "typescript", "react", "redux", "angular", "vue",
    "node.js", "node", "express", "flask", "django", "spring", "mongodb", "sql", "nosql",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket",
    "jenkins", "ci/cd", "firebase", "redis", "nginx", "apache", "lambda", "cloudwatch",
    "s3", "ec2", "rds", "dynamodb", "power bi", "tableau", "excel", "quickbooks",
    "bloomberg terminal", "bloomberg", "salesforce crm", "sap", "oracle db", "mysql",
    "postgresql", "c++", "c#", "ruby", "php", "swift", "kotlin", "go", "rust",
    "html", "css", "sass", "scss", "bootstrap", "tailwind", "webpack", "babel",
    "photoshop", "illustrator", "figma", "sketch", "adobe xd", "indesign"
}

def extract_experience_section(text):
    """
    Extract the work experience section from resume text.
    """
    # Common section headers for work experience
    experience_patterns = [
        r'(?:^|\n)[\s]*(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT\s+HISTORY|WORK\s+HISTORY|CAREER\s+HISTORY)[\s]*:?\s*\n',
    ]
    
    # Common section headers that come AFTER experience
    next_section_patterns = [
        r'(?:^|\n)[\s]*(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|CERTIFICATES|LANGUAGES|INTERESTS|REFERENCES|PUBLICATIONS|AWARDS|VOLUNTEER)[\s]*:?\s*\n',
    ]
    
    experience_section = None
    
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_pos = match.end()
            
            # Find where experience section ends (next section starts)
            end_pos = len(text)
            for end_pattern in next_section_patterns:
                end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
                if end_match:
                    end_pos = start_pos + end_match.start()
                    break
            
            experience_section = text[start_pos:end_pos]
            break
    
    return experience_section

def extract_job_entries(experience_text):
    """
    Parse individual job entries from experience section.
    Each entry typically has: Job Title, Company Name, Duration, and bullet points.
    
    Common formats:
    1. Job Title — Company Name (Duration)
    2. Job Title | Company Name | Duration
    3. Job Title at Company Name (Duration)
    """
    if not experience_text:
        return []
    
    job_entries = []
    
    # Most common pattern: "Job Title — Company Name (Date Range)"
    # Matches formats like:
    # - Software Developer — CloudWorks Technologies (Apr 2022 – Present)
    # - Junior Backend Developer — NovaCode Solutions (Jan 2021 – Mar 2022)
    main_pattern = r'([A-Z][^\n—\-]+?)\s*[—\-–]\s*([A-Z][^\n(]+?)\s*\(([^)]+)\)'
    
    matches = re.finditer(main_pattern, experience_text)
    
    for match in matches:
        job_title = match.group(1).strip()
        company = match.group(2).strip()
        duration = match.group(3).strip()
        
        # Validate that duration contains dates
        date_indicators = ['20\d{2}', 'present', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        duration_lower = duration.lower()
        
        if any(re.search(indicator, duration_lower) for indicator in date_indicators):
            job_entries.append({
                'job_title': job_title,
                'company': company,
                'duration': duration,
                'raw_text': match.group(0)
            })
    
    # Fallback: Alternative patterns
    if not job_entries:
        # Try pipe-separated format: "Job Title | Company Name | Duration"
        pipe_pattern = r'([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+)'
        matches = re.finditer(pipe_pattern, experience_text)
        
        for match in matches:
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()
            part3 = match.group(3).strip()
            
            # Check which part has dates
            date_indicators = ['20\d{2}', 'present', 'jan', 'feb', 'mar', 'apr', 'may', 'jun']
            
            if any(re.search(indicator, part3.lower()) for indicator in date_indicators):
                # part3 is duration, determine job title vs company
                job_title_keywords = [
                    'manager', 'director', 'engineer', 'developer', 'analyst', 'designer',
                    'specialist', 'coordinator', 'consultant', 'representative', 'officer',
                    'supervisor', 'lead', 'architect', 'associate', 'intern', 'assistant'
                ]
                
                part1_has_title = any(kw in part1.lower() for kw in job_title_keywords)
                
                if part1_has_title:
                    job_title = part1
                    company = part2
                else:
                    job_title = part2
                    company = part1
                
                job_entries.append({
                    'job_title': job_title,
                    'company': company,
                    'duration': part3,
                    'raw_text': match.group(0)
                })
    
    # Another fallback: Multi-line format
    if not job_entries:
        lines = experience_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('-') or line.startswith('•'):
                i += 1
                continue
            
            # Check if line contains date pattern and parentheses
            if '(' in line and ')' in line:
                # Extract content in parentheses (likely duration)
                paren_match = re.search(r'\(([^)]+)\)', line)
                if paren_match:
                    duration = paren_match.group(1)
                    date_indicators = ['20\d{2}', 'present', 'jan', 'feb', 'mar', 'apr', 'may', 'jun']
                    
                    if any(re.search(indicator, duration.lower()) for indicator in date_indicators):
                        # Extract job title and company (before the parentheses)
                        before_paren = line[:paren_match.start()].strip()
                        
                        # Split by dash or at
                        parts = re.split(r'\s*[—\-–]\s*|\s+at\s+', before_paren, maxsplit=1)
                        
                        if len(parts) == 2:
                            job_entries.append({
                                'job_title': parts[0].strip(),
                                'company': parts[1].strip(),
                                'duration': duration,
                                'raw_text': line
                            })
            
            i += 1
    
    return job_entries

def clean_company_name(company):
    """Clean and validate company name."""
    if not company:
        return None
    
    company = company.strip()
    
    # Remove common separators and extra info
    company = re.split(r'\s*[|\-–]\s*(?:Remote|Hybrid|On-site|Full-time|Part-time|Contract)', company, flags=re.IGNORECASE)[0]
    company = re.split(r'\s*[,;]\s*(?:Remote|Hybrid|On-site|Full-time|Part-time|Contract)', company, flags=re.IGNORECASE)[0]
    
    # Remove location info (City, State format)
    company = re.sub(r',\s*[A-Z]{2}$', '', company)  # Remove state codes
    company = re.sub(r',\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', '', company)  # Remove city names
    
    # Remove trailing punctuation
    company = re.sub(r'[,;:\-–]+$', '', company).strip()
    
    company_lower = company.lower()
    
    # Check if it's a tech tool, not a company
    if company_lower in TECH_TOOLS:
        return None
    
    # Check if it's too short or too long
    if len(company) < 2 or len(company) > 60:
        return None
    
    # Check if it starts with a capital letter
    if not (company[0].isupper() or company[0].isdigit()):
        return None
    
    return company

def extract_job_org_relations(text):
    """
    Extract job-organization relationships by focusing on the experience section.
    """
    relations = []
    
    # Step 1: Extract experience section
    experience_section = extract_experience_section(text)
    
    if not experience_section:
        # Fallback: treat entire text as experience section
        experience_section = text
    
    # Step 2: Parse individual job entries
    job_entries = extract_job_entries(experience_section)
    
    # Step 3: Extract job title and company from each entry
    for entry in job_entries:
        job_title = entry.get('job_title', '').strip()
        company = entry.get('company', '').strip()
        
        # Clean and validate company name
        company = clean_company_name(company)
        
        if job_title and company:
            # Format the relationship
            relations.append(f"{job_title} at {company}")
    
    # Fallback: If no structured entries found, try simple pattern matching
    if not relations:
        # Look for "at Company" or "@ Company" patterns
        at_pattern = r'(?:at|@)\s+([A-Z][a-zA-Z\s&.\'-]+?)(?:\s|,|\.|$|\|)'
        matches = re.findall(at_pattern, experience_section if experience_section else text)
        
        for match in matches:
            company = clean_company_name(match)
            if company:
                relations.append(f"Experience at {company}")
    
    return relations if relations else ["No clear job-org relationships found"]

# Skills extraction
skills_pool = {
    "TECH": [
        # INFORMATION-TECHNOLOGY
        "python","java","c++","c#","javascript","react","node.js","sql","html","css",
        "typescript","flask","django","spring","software development","data analysis",
        "machine learning","ai","pandas","numpy","git",

        # ENGINEERING
        "autocad","solidworks","matlab","design analysis","project management",
        "mechanical systems","civil design","electrical circuits","simulation","blueprint reading",

        # DIGITAL-MEDIA
        "social media","seo","content creation","copywriting","google ads","facebook ads",
        "email marketing","campaign management","analytics","branding",

        # DESIGNER
        "photoshop","illustrator","figma","adobe xd","ui design","ux design","wireframing",
        "logo design","graphic design","typography","prototyping","creativity",

        # AUTOMOBILE
        "vehicle maintenance","mechanical repair","diagnostics","automotive systems",
        "engine tuning","electrical systems","safety inspection","parts replacement",

        # AVIATION
        "flight operations","navigation","safety procedures","aircraft maintenance",
        "air traffic communication","crew management","aviation regulations",
    ],

    "BUSINESS": [
        # BUSINESS-DEVELOPMENT
        "sales strategy","client relations","market research","partnerships","lead generation",
        "b2b","crm","pipeline management","business strategy","negotiation",

        # FINANCE
        "accounting","budgeting","financial analysis","investment","forecasting","auditing",
        "taxation","risk management","bookkeeping","excel","financial modeling",

        # ACCOUNTANT
        "financial reporting","tax preparation","reconciliation","financial statements",
        "accounting software",

        # BANKING
        "customer service","loan processing","credit analysis","cash handling",
        "financial advisory","branch operations","risk assessment","compliance",

        # SALES
        "customer relationship management","closing deals","cold calling","presentation",
        "sales forecasting","upselling",

        # CONSULTANT
        "business analysis","project management","strategy","stakeholder management",
        "data analysis","problem solving","process improvement",

        # APPAREL
        "fashion design","textile","sewing","pattern making","trend analysis","styling",
        "merchandising","fabric selection","retail management",

        # HR
        "recruitment","interviewing","onboarding","employee relations","hr policies",
        "payroll","talent acquisition","performance management","training",
        "benefits administration",
    ],

    "HEALTH_FITNESS": [
        # HEALTHCARE
        "patient care","diagnosis","medical records","phlebotomy","emergency response",
        "clinical procedures","first aid","surgery assistance","public health","pharmacology",

        # FITNESS
        "personal training","nutrition","fitness assessment","strength training","cardio",
        "injury prevention","yoga","exercise programming","motivation","client coaching",
    ],

    "EDUCATION": [
        # TEACHER
        "lesson planning","curriculum development","classroom management","grading",
        "student engagement","communication","mentoring","public speaking",
        "education technology",

        # ARTS
        "painting","drawing","illustration","creative writing","music composition",
        "photography","editing","storytelling","animation","art direction",

        # ADVOCATE
        "legal research","drafting","litigation","contracts","compliance","corporate law",
        "intellectual property","negotiation","case management","advocacy",
    ],

    "SERVICE": [
        # CHEF
        "cooking","menu planning","food safety","recipe development","plating","baking",
        "inventory management","teamwork","sanitation","knife skills",

        # BPO
        "inbound calls","outbound calls","ticketing","crm software","technical support",
        "time management","problem solving","escalation handling",

        # PUBLIC-RELATIONS
        "media relations","press release","branding","crisis communication","copywriting",
        "public speaking","event coordination","marketing communication",
    ],

    "LABOR_TRADES": [
        # CONSTRUCTION
        "site supervision","blueprint reading","safety management","quantity surveying",
        "project scheduling","civil works","materials management","contract management",

        # AGRICULTURE
        "crop management","soil science","irrigation","harvesting","farm equipment",
        "organic farming","agribusiness","animal husbandry","pesticide control","planting",
    ]
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