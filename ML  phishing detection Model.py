
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lime.lime_text import LimeTextExplainer
import numpy as np

# Loading and Preprocessing of Data

df = pd.read_csv("final_combined_phishing_dataset.csv")
df = df.dropna(subset=['subject', 'body', 'label'])
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
df = df[["text", "label"]]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df.text.values, df.label.values, test_size=0.2, stratify=df.label, random_state=42
)

# Tokenization with BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=256, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = EmailDataset(train_texts, train_labels)
test_dataset = EmailDataset(test_texts, test_labels)


# Combining BERT + LSTM architecture for a hybrid Model

class BERT_LSTM(nn.Module):
    def __init__(self):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm_output, _ = self.lstm(bert_output)
        pooled = lstm_output[:, -1, :]
        out = self.dropout(pooled)
        return torch.sigmoid(self.fc(out))




# Explainability using LIME

class_names = ["legit", "phish"]
def predict_prob(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = model(**inputs).cpu().numpy()
    return np.hstack([1 - probs, probs])

explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(test_texts[0], predict_prob, num_features=10)
exp.show_in_notebook()

