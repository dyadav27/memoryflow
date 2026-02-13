# MemoryFlow

**MemoryFlow** is a persistent memory system designed for autonomous AI agents. It enables long-term recall by classifying, filtering, and ranking user interactions while maintaining performance under heavy noise.

## 🚀 Overview

MemoryFlow ensures AI agents don't "forget" by combining structured SQL storage with semantic vector search. Key capabilities include:

* **Intent Classification:** Categorizes inputs into facts, preferences, instructions, or constraints.
* **Semantic Retrieval:** Uses FAISS for high-speed vector similarity searches.
* **Dynamic Ranking:** Applies a salience formula based on confidence, memory type, and recency.
* **Noise Resilience:** Filters out irrelevant data and rejects low-confidence inputs.
* **Persistence:** Durable storage using SQLite.

---

## 🏗️ Architecture

The system follows a linear pipeline to ensure data integrity and retrieval accuracy:

1. **User Input** → Processed by **DistilBERT Classifier**.
2. **Sentence Splitting** → Filtered for noise and relevance.
3. **Storage** → Saved in **SQLite** and indexed via **FAISS Vector Index**.
4. **Salience Ranking** → Calculated as:
`Score = (Confidence × Type Weight) / Age`
5. **Retrieval** → Fetches top-ranked relevant memories.
6. **Summarization** → Context is refined via **BART Summarizer**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** Transformers (Hugging Face)
* **Models:** DistilBERT (Classification), MiniLM (Embeddings), BART (Summarization)
* **Vector Database:** FAISS
* **Database:** SQLite
* **Frameworks:** PyTorch, SentenceTransformers

---

## 📦 Installation

Ensure you have Python 3.8+ installed, then run:

```bash
pip install faiss-cpu sentence-transformers transformers datasets torch

```
---

## 🚦 Getting Started

### 1. Train the Classifier

Fine-tune the DistilBERT model to recognize different types of user input:

```bash
python train_classifier.py

```

### 2. Run Stress Test

Validate the system's ability to recall specific facts amidst 1,000+ noisy data points:

```bash
python stress_test.py

```

---

## 📊 Stress Test Results

After injecting **1,000+ random/noisy memories**, the system maintains a 100% retrieval rate for core user attributes:

| Query | System Output |
| --- | --- |
| **Name** | "My name is Darshan" |
| **Language** | "I prefer Hindi" |
| **Location** | "I live in Mumbai" |
| **Tech Interest** | "My favorite tech is AI" |

---

## 💡 Example Usage

**Query:** *"What do you know about me?"*

**Response:**

> * Your name is Darshan.
> * You prefer communicating in Hindi.
> * You are based in Mumbai.
> * You have a keen interest in AI technology.
> 
> 

---

## 👤 Author

**Darshan Yadav**
**Sarang Patil**
**Pranav Patil**

