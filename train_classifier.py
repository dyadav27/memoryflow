from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

NUM_LABELS = 6

print("Loading CSV dataset...")

dataset = load_dataset(
    "csv",
    data_files="universal_training_data_70k.csv"
)["train"]

print("Total samples:", len(dataset))

# ============================
# Tokenizer
# ============================

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.remove_columns(["text"])

data_collator = DataCollatorWithPadding(tokenizer)

# ============================
# Model
# ============================

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=NUM_LABELS
)

# ============================
# Training Arguments
# ============================

training_args = TrainingArguments(
    output_dir="./classifier",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=200,
    save_strategy="epoch",
    report_to="none"
)

# ============================
# Trainer (NO tokenizer param!)
# ============================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("Starting training...")

trainer.train()

print("Saving model...")

trainer.save_model("./classifier")
tokenizer.save_pretrained("./classifier")

print("\n✅ TRAINING COMPLETE")
