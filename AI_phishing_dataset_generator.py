import openai
import openai
import os
import random
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# loads environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# phishing prompt templates
phishing_templates = [
    "Generate a phishing email pretending to be from a bank, asking the user to verify their account.",
    "Write a phishing email that impersonates a company requesting login credentials due to suspicious activity.",
    "Generate a phishing email that uses urgent language to get the user to click a malicious link.",
    "Create a phishing email pretending to be from a government agency threatening legal action.",
    "Generate a phishing email disguised as a job offer asking for personal information.",
    "Write a phishing email claiming the user has won a prize and needs to provide their details to claim it.",
    "Create a phishing email posing as a tech support warning the user of a virus and prompting them to install software."
]

legitimate_templates = [
    "Write a legitimate bank notification about a new feature on the mobile app.",
    "Generate a genuine company newsletter informing about upcoming product updates.",
    "Create an official email from a university about an upcoming academic event.",
    "Write a legitimate email confirming a recent online order from an e-commerce site.",
    "Generate an email from a government agency about community safety tips.",
    "Create a legitimate email from an HR department sharing employee benefit updates."
]

def generate_email(prompt, label, max_tokens=200, temperature=0.8):
    """Generates a single email using the OpenAI API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are an expert email writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response['choices'][0]['message']['content'].strip()
        return {"email_text": text, "label": label}
    except Exception as e:
        print(f"Error generating email: {e}")
        return None

def generate_dataset(n_phishing=50, n_legit=50):
    """Generates a dataset with phishing and legitimate emails."""
    data = []
    
    print("[+] Generating phishing emails...")
    for _ in tqdm(range(n_phishing)):
        prompt = random.choice(phishing_templates)
        sample = generate_email(prompt, label=1)
        if sample:
            data.append(sample)

    print("[+] Generating legitimate emails...")
    for _ in tqdm(range(n_legit)):
        prompt = random.choice(legitimate_templates)
        sample = generate_email(prompt, label=0)
        if sample:
            data.append(sample)

    df = pd.DataFrame(data)
    df.to_csv("generated_phishing_dataset.csv", index=False)
    print("[✓] Dataset saved as generated_phishing_dataset.csv")

if __name__ == "__main__":
    generate_dataset(n_phishing=100, n_legit=100)  


