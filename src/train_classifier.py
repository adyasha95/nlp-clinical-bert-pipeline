import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert-base-uncased")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

df = pd.read_csv("data/synthetic_clinical_notes.csv")
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(df.label.unique()))

training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    evaluation_strategy="epoch",
    learning_rate=args.lr,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("models/best_model")
tokenizer.save_pretrained("models/best_model")
