import datetime
from openai import OpenAI
import os
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time

# loads environment variables from .env
load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

top_industries = [
    {"name": "Insurance", "weight": 25},
    {"name": "Finance", "weight": 25},
    {"name": "Healthcare", "weight": 20},
    {"name": "Law", "weight": 15},
    {"name": "Transportation", "weight": 15},
]

# Top 5 most impersonated brands (based on Egress report)
top_brands = [
    {"name": "Microsoft", "domain": "microsoft.com", "weight": 30},
    {"name": "DocuSign", "domain": "docusign.com", "weight": 25},
    {"name": "PayPal", "domain": "paypal.com", "weight": 20},
    {"name": "DHL", "domain": "dhl.com", "weight": 15},
    {"name": "Facebook", "domain": "facebook.com", "weight": 10}
]

# Top 5 most targeted job titles (based on Egress report)
top_job_titles = [
    {"title": "CEO", "weight": 30},
    {"title": "CFO", "weight": 25},
    {"title": "CPO", "weight": 15},
    {"title": "CISO", "weight": 15},
    {"title": "CRO", "weight": 15}
]

# Payload types (link and attachment - removing QR codes)
payload_types = [
    {"type": "link", "weight": 70},
    {"type": "attachment", "weight": 30}
]

def generate_spoofed_url(domain):
    technique = random.choice(["typosquatting", "subdomain_spoofing", "homoglyphs"])
    if technique == "typosquatting":
        typo_options = [
            domain.replace("a", "e", 1) if "a" in domain else domain,
            domain.replace("o", "0", 1) if "o" in domain else domain,
            domain.replace("i", "1", 1) if "i" in domain else domain,
            domain.replace("l", "1", 1) if "l" in domain else domain,
            domain.replace("s", "5", 1) if "s" in domain else domain,
            domain.replace("n", "m", 1) if "n" in domain else domain,
            domain.replace(".", "-.", 1),
            domain[:len(domain)-4] + ".org" if domain.endswith(".com") else domain,
        ]
        return random.choice(typo_options), "typosquatting"
    elif technique == "subdomain_spoofing":
        prefix = random.choice(["secure", "account", "login", "signin", "verify", "service"])
        return f"{prefix}.{domain.split('.')[0]}.phishing-domain.com", "subdomain_spoofing"
    else:
        homoglyphs = {'a': 'а','e': 'е','o': 'о','p': 'р','c': 'с','x': 'х','i': 'і'}
        spoofed = domain
        chars_to_replace = min(2, sum(1 for c in domain if c in homoglyphs))
        replaced = 0
        for char in homoglyphs:
            if char in spoofed and replaced < chars_to_replace:
                index = spoofed.find(char)
                spoofed = spoofed[:index] + homoglyphs[char] + spoofed[index+1:]
                replaced += 1
        return spoofed, "homoglyphs"

def generate_attachment_name(company_name, industry):
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    industry_attachments = {
        "Insurance": [f"Insurance_Policy_Update_{date_str}.pdf", f"Claim_Form_{random.randint(10000, 99999)}.pdf", f"Policy_Renewal_{date_str}.pdf", f"Insurance_Premium_Statement.xlsx", f"Coverage_Changes_{date_str}.docx"],
        "Finance": [f"Financial_Statement_{date_str}.pdf", f"Transaction_Report_{date_str}.xlsx", f"Investment_Update_{date_str}.pdf", f"Tax_Document_{random.randint(1000, 9999)}.pdf", f"Account_Statement_{date_str}.pdf"],
        "Healthcare": [f"Medical_Record_Update.pdf", f"Patient_Information_{random.randint(10000, 99999)}.pdf", f"Insurance_Verification_Form.pdf", f"Healthcare_Policy_Update.docx", f"Lab_Results_{date_str}.pdf"],
        "Law": [f"Legal_Document_{random.randint(1000, 9999)}.pdf", f"Case_Update_{date_str}.pdf", f"Contract_Review_{date_str}.docx", f"Legal_Notice_{random.randint(1000, 9999)}.pdf", f"Client_Agreement.pdf"],
        "Transportation": [f"Shipping_Manifest_{date_str}.pdf", f"Delivery_Confirmation_{random.randint(10000, 99999)}.pdf", f"Tracking_Update_{random.randint(10000, 99999)}.pdf", f"Transportation_Invoice_{date_str}.xlsx", f"Shipment_Details_{date_str}.pdf"]
    }
    default_attachments = [f"{company_name}_Invoice_{date_str}.pdf", f"{company_name}_Statement_{date_str}.pdf", f"{company_name}_Receipt_{random.randint(10000, 99999)}.pdf", f"{company_name}_Document_Scan.pdf", f"{company_name}_Security_Update.pdf"]
    return random.choice(industry_attachments.get(industry, default_attachments))

def select_weighted_item(items):
    weights = [item.get("weight", 1) for item in items]
    return random.choices(items, weights=weights, k=1)[0]

def generate_phishing_email(prompt, max_tokens=350, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are generating examples of phishing emails for security research purposes only."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
        return {"email_text": text}
    except Exception as e:
        print(f"Error generating phishing email: {e}")
        return None

def generate_legitimate_email(prompt, max_tokens=350, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are generating examples of legitimate company emails for security research purposes."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        print(f"Error generating legitimate email: {e}")
        return None

def extract_email_parts(email_text):
    subject = ""
    body = email_text
    lines = email_text.split('\n')
    for i, line in enumerate(lines):
        if line.lower().startswith("subject:"):
            subject = line[8:].strip()
            body = '\n'.join(lines[i+1:]).strip()
            break
    return subject, body

def generate_dataset(n_phishing=100, n_legit=50, output_path="phishing_dataset.csv"):
    data = []
    print("[+] Generating phishing emails based on Egress report findings...")
    for _ in tqdm(range(n_phishing)):
        try:
            industry = select_weighted_item(top_industries)["name"]
            brand = select_weighted_item(top_brands)
            job_title = select_weighted_item(top_job_titles)["title"]
            payload = select_weighted_item(payload_types)["type"]
            first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emma", "Robert", "Lisa"]
            last_names = ["Smith", "Jones", "Brown", "Johnson", "Garcia", "Miller", "Davis", "Wilson"]
            target_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            company_suffixes = {
                "Insurance": ["Insurance", "Assurance", "Coverage", "Protect"],
                "Finance": ["Financial", "Capital", "Invest", "Banking"],
                "Healthcare": ["Health", "Medical", "Care", "Wellness"],
                "Law": ["Legal", "Law", "Partners", "Associates"],
                "Transportation": ["Transport", "Logistics", "Shipping", "Freight"]
            }
            target_company = f"{random.choice(['Global', 'United', 'Premier', 'Elite', 'Advanced'])} {random.choice(company_suffixes.get(industry, ['Company']))}"
            spoofed_url, phishing_technique = generate_spoofed_url(brand["domain"])
            attachment_name = generate_attachment_name(brand["name"], industry) if payload == "attachment" else None
            payload_instruction = f"Include a link that the user needs to click to resolve an urgent issue with their {brand['name']} account." if payload == "link" else f"Mention an important attachment ({attachment_name}) that the user needs to review immediately."
            prompt = f"""Generate an email that appears to be legitimate from \"{brand['name']}\" targeted at a {job_title} in the {industry} industry.
The email should be addressed to \"{target_name}\", {job_title} at {target_company}.
{payload_instruction}
The purpose is to create a sense of urgency and convince a high-level executive to take immediate action.
"""
            phishing_email = generate_phishing_email(prompt)
            if phishing_email:
                subject, body = extract_email_parts(phishing_email["email_text"])
                data.append({
                    "email_type": "phishing",
                    "subject": subject,
                    "body": body,
                    "phishing_technique": phishing_technique,
                    "target_name": target_name,
                    "target_job_title": job_title,
                    "target_company": target_company,
                    "spoofed_url": spoofed_url,
                    "attachment_name": attachment_name,
                    "industry": industry
                })
            time.sleep(2)  # Adding sleep to avoid rate limiting
        except Exception as e:
            print(f"Error generating phishing email: {e}")
    
    print("[+] Generating legitimate emails...")
    for _ in tqdm(range(n_legit)):
        try:
            company = select_weighted_item(top_brands)["name"]
            job_title = select_weighted_item(top_job_titles)["title"]
            first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emma", "Robert", "Lisa"]
            last_names = ["Smith", "Jones", "Brown", "Johnson", "Garcia", "Miller", "Davis", "Wilson"]
            target_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            prompt = f"""Generate an email from the HR department of {company} addressed to {target_name}, {job_title}. It should be a professional and polite email inviting them to a meeting for a job interview.
"""
            legit_email = generate_legitimate_email(prompt)
            if legit_email:
                subject, body = extract_email_parts(legit_email)
                data.append({
                    "email_type": "legitimate",
                    "subject": subject,
                    "body": body,
                    "phishing_technique": None,
                    "target_name": target_name,
                    "target_job_title": job_title,
                    "target_company": company,
                    "spoofed_url": None,
                    "attachment_name": None,
                    "industry": None
                })
            time.sleep(2)  # Adding sleep to avoid rate limiting
        except Exception as e:
            print(f"Error generating legitimate email: {e}")
    
    print("[+] Saving dataset to CSV...")
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset saved to {output_path}")

generate_dataset()

