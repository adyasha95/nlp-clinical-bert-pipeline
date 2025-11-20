import pandas as pd
import random

conditions = [
    "mild fever and fatigue",
    "shortness of breath",
    "abdominal pain",
    "stable vital signs",
    "reports dizziness and nausea",
    "nonadherence to medication",
    "infection risk due to wound redness",
    "requires follow-up consultation"
]

labels = [
    "stable_condition",
    "medication_nonadherence",
    "infection_risk",
    "follow_up_required"
]

def generate_synthetic_note():
    template = random.choice(conditions)
    label = random.choice(labels)
    return template, label

notes = []
for _ in range(500):  # generate 500 synthetic notes
    text, label = generate_synthetic_note()
    notes.append({"text": text, "label": label})

df = pd.DataFrame(notes)
df.to_csv("data/synthetic_clinical_notes.csv", index=False)

print("Synthetic dataset created!")
