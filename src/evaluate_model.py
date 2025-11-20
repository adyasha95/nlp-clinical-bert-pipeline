import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

df = pd.read_csv("data/synthetic_clinical_notes.csv")
dataset = Dataset.from_pandas(df)

labels = sorted(df.label.unique())
label2id = {l:i for i,l in enumerate(labels)}
id2label = {i:l for l,i in label2id.items()}

model_path = "models/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    output = model(**tokens)
    return output.logits.argmax(-1).item()

preds = [predict(t) for t in df.text]
true = [label2id[l] for l in df.label]

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
