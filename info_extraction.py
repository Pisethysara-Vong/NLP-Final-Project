import re

# Extract experience section
def extract_experience_section(text):
    experience_patterns = [
        r'(?:^|\n)[\s]*(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT\s+HISTORY|WORK\s+HISTORY|CAREER\s+HISTORY)[\s]*:?\s*\n',
    ]
    
    next_section_patterns = [
        r'(?:^|\n)[\s]*(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|CERTIFICATES|LANGUAGES|INTERESTS|REFERENCES|PUBLICATIONS|AWARDS|VOLUNTEER)[\s]*:?\s*\n',
    ]
    
    experience_section = None
    
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_pos = match.end()
            end_pos = len(text)
            for end_pattern in next_section_patterns:
                end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
                if end_match:
                    end_pos = start_pos + end_match.start()
                    break
            experience_section = text[start_pos:end_pos]
            break
    
    return experience_section

# Job entry extraction
def extract_job_entries(experience_text):
    if not experience_text:
        return []
    
    job_entries = []
    
    main_pattern = r'([A-Z][^\n—\-]+?)\s*[—\-–]\s*([A-Z][^\n(]+?)\s*\(([^)]+)\)'
    matches = re.finditer(main_pattern, experience_text)
    
    for match in matches:
        job_title = match.group(1).strip()
        company = match.group(2).strip()
        duration = match.group(3).strip()

        date_indicators = ['20\d{2}', 'present', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if any(re.search(indicator, duration.lower()) for indicator in date_indicators):
            job_entries.append({
                'job_title': job_title,
                'company': company,
                'duration': duration,
                'raw_text': match.group(0)
            })

    return job_entries

# Job-Organization relationship extraction
def extract_job_org_relations(text):
    relations = []
    experience_section = extract_experience_section(text)

    if not experience_section:
        experience_section = text

    job_entries = extract_job_entries(experience_section)

    for entry in job_entries:
        job_title = entry.get('job_title', '').strip()
        company = entry.get('company', '').strip()

        if job_title and company:
            relations.append(f"{job_title} at {company}")

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

def extract_skills(text):
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
