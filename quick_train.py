<<<<<<< HEAD
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------------------------------
# Load CSV
# ---------------------------------

df = pd.read_csv("train.csv")

label_map = {
    "preference":0,
    "constraint":1,
    "fact":2,
    "instruction":3,
    "ignore":4,
    "command":5
}

df["labels"] = df["label"].map(label_map)
df = df[["text","labels"]]

dataset = Dataset.from_pandas(df)

# ---------------------------------
# Tokenizer
# ---------------------------------

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)

# VERY IMPORTANT
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# ---------------------------------
# Model
# ---------------------------------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# ---------------------------------
# Training
# ---------------------------------

args = TrainingArguments(
    output_dir="./classifier",
    per_device_train_batch_size=4,
    num_train_epochs=8,
    logging_steps=2,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./classifier")
tokenizer.save_pretrained("./classifier")
=======
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ---------------------------------
# Load CSV
# ---------------------------------

df = pd.read_csv("train.csv")

label_map = {
    "preference":0,
    "constraint":1,
    "fact":2,
    "instruction":3,
    "ignore":4,
    "command":5
}

df["labels"] = df["label"].map(label_map)
df = df[["text","labels"]]

dataset = Dataset.from_pandas(df)

# ---------------------------------
# Tokenizer
# ---------------------------------

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)

# VERY IMPORTANT
dataset = dataset.remove_columns(["text"])
dataset.set_format("torch")

# ---------------------------------
# Model
# ---------------------------------

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# ---------------------------------
# Training
# ---------------------------------

args = TrainingArguments(
    output_dir="./classifier",
    per_device_train_batch_size=4,
    num_train_epochs=8,
    logging_steps=2,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./classifier")
tokenizer.save_pretrained("./classifier")
>>>>>>> ddf6092 (Initial MemoryFlow submission)
