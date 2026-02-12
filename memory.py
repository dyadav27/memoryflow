<<<<<<< HEAD
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from db import insert_memory, load_memories, update_memory, delete_memory

# ===============================
# Labels
# ===============================

LABEL_MAP = {
    "LABEL_0": "preference",
    "LABEL_1": "constraint",
    "LABEL_2": "fact",
    "LABEL_3": "instruction",
    "LABEL_4": "ignore",
    "LABEL_5": "command"
}

PROFILE_TYPES = ["fact", "preference"]

# ===============================
# Models
# ===============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = pipeline("text-classification", model="./classifier", top_k=1)

# ===============================
# FAISS
# ===============================

DIM = 384
index = faiss.IndexFlatIP(DIM)

# ===============================
# Memory
# ===============================

memories = []
vectors = []
profile = []
turn = 0

# ===============================
# Helpers
# ===============================

def next_turn():
    global turn
    turn += 1
    return turn

def embed(texts):
    return embedder.encode(texts, normalize_embeddings=True).astype("float32")

def is_noise(t):
    if len(t.split()) < 3:
        return True
    if t.startswith("i saw"):
        return True
    if re.search(r"[a-z]{12,}", t):
        return True
    return False

def is_identity_query(q):
    return any(x in q.lower() for x in [
        "about me", "who am i", "my info", "what do you know"
    ])

def split_sentences(text):
    parts = re.split(r"[.!?;\n,]", text.lower())
    return [p.strip() for p in parts if len(p.strip()) > 3]

# ===============================
# Load Existing
# ===============================

def load_existing():

    rows = load_memories()
    if not rows:
        return

    texts = []

    for r in rows:
        mem = {
            "id": r[0],
            "content": r[1],
            "type": r[2],
            "confidence": r[3],
            "turn": r[4],
            "last_used": r[5]
        }

        memories.append(mem)

        if mem["type"] in PROFILE_TYPES:
            profile.append(mem)

        texts.append(mem["content"])

    vecs = embed(texts)
    vectors.extend(vecs)
    index.add(vecs)

load_existing()

# ===============================
# Classification
# ===============================

def detect_type(text):
    return LABEL_MAP[classifier(text)[0][0]["label"]]

# ===============================
# Add Memory
# ===============================

def add_memory(text, current_turn):

    parts = split_sentences(text)

    for p in parts:

        if is_noise(p):
            continue

        if any(m["content"] == p for m in memories):
            continue

        t = detect_type(p)

        if t in ["ignore", "command"]:
            continue

        vec = embed([p])[0]

        row_id = insert_memory(p, t, 0.9, current_turn, current_turn)

        mem = {
            "id": row_id,
            "content": p,
            "type": t,
            "confidence": 0.9,
            "turn": current_turn,
            "last_used": current_turn
        }

        memories.append(mem)
        vectors.append(vec)
        index.add(np.array([vec]))

        if t in PROFILE_TYPES:
            profile.append(mem)

# ===============================
# Retrieve
# ===============================

def retrieve_memories(query, k=5):

    if is_identity_query(query):
        return profile[:k]

    if not memories:
        return []

    q = embed([query])

    D, I = index.search(q, k*3)

    results = []

    for sim, idx in zip(D[0], I[0]):

        if sim < 0.55:
            continue

        m = memories[idx]

        m["confidence"] = min(m["confidence"] + 0.05, 1.0)
        m["last_used"] = turn

        update_memory(m["id"], m["confidence"], m["last_used"])

        results.append(m)

    return results[:k]

# ===============================
# Decay
# ===============================

def decay_memories():

    alive_m = []
    alive_v = []

    for m,v in zip(memories,vectors):

        m["confidence"] *= 0.995

        if m["confidence"] > 0.2:
            alive_m.append(m)
            alive_v.append(v)
            update_memory(m["id"], m["confidence"], m["last_used"])
        else:
            delete_memory(m["id"])

    memories[:] = alive_m
    vectors[:] = alive_v

    index.reset()
    if vectors:
        index.add(np.vstack(vectors))
=======
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from db import insert_memory, load_memories, update_memory, delete_memory

# ===============================
# Labels
# ===============================

LABEL_MAP = {
    "LABEL_0": "preference",
    "LABEL_1": "constraint",
    "LABEL_2": "fact",
    "LABEL_3": "instruction",
    "LABEL_4": "ignore",
    "LABEL_5": "command"
}

PROFILE_TYPES = ["fact", "preference"]

# ===============================
# Models
# ===============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = pipeline("text-classification", model="./classifier", top_k=1)

# ===============================
# FAISS
# ===============================

DIM = 384
index = faiss.IndexFlatIP(DIM)

# ===============================
# Memory
# ===============================

memories = []
vectors = []
profile = []
turn = 0

# ===============================
# Helpers
# ===============================

def next_turn():
    global turn
    turn += 1
    return turn

def embed(texts):
    return embedder.encode(texts, normalize_embeddings=True).astype("float32")

def is_noise(t):
    if len(t.split()) < 3:
        return True
    if t.startswith("i saw"):
        return True
    if re.search(r"[a-z]{12,}", t):
        return True
    return False

def is_identity_query(q):
    return any(x in q.lower() for x in [
        "about me", "who am i", "my info", "what do you know"
    ])

def split_sentences(text):
    parts = re.split(r"[.!?;\n,]", text.lower())
    return [p.strip() for p in parts if len(p.strip()) > 3]

# ===============================
# Load Existing
# ===============================

def load_existing():

    rows = load_memories()
    if not rows:
        return

    texts = []

    for r in rows:
        mem = {
            "id": r[0],
            "content": r[1],
            "type": r[2],
            "confidence": r[3],
            "turn": r[4],
            "last_used": r[5]
        }

        memories.append(mem)

        if mem["type"] in PROFILE_TYPES:
            profile.append(mem)

        texts.append(mem["content"])

    vecs = embed(texts)
    vectors.extend(vecs)
    index.add(vecs)

load_existing()

# ===============================
# Classification
# ===============================

def detect_type(text):
    return LABEL_MAP[classifier(text)[0][0]["label"]]

# ===============================
# Add Memory
# ===============================

def add_memory(text, current_turn):

    parts = split_sentences(text)

    for p in parts:

        if is_noise(p):
            continue

        if any(m["content"] == p for m in memories):
            continue

        t = detect_type(p)

        if t in ["ignore", "command"]:
            continue

        vec = embed([p])[0]

        row_id = insert_memory(p, t, 0.9, current_turn, current_turn)

        mem = {
            "id": row_id,
            "content": p,
            "type": t,
            "confidence": 0.9,
            "turn": current_turn,
            "last_used": current_turn
        }

        memories.append(mem)
        vectors.append(vec)
        index.add(np.array([vec]))

        if t in PROFILE_TYPES:
            profile.append(mem)

# ===============================
# Retrieve
# ===============================

def retrieve_memories(query, k=5):

    if is_identity_query(query):
        return profile[:k]

    if not memories:
        return []

    q = embed([query])

    D, I = index.search(q, k*3)

    results = []

    for sim, idx in zip(D[0], I[0]):

        if sim < 0.55:
            continue

        m = memories[idx]

        m["confidence"] = min(m["confidence"] + 0.05, 1.0)
        m["last_used"] = turn

        update_memory(m["id"], m["confidence"], m["last_used"])

        results.append(m)

    return results[:k]

# ===============================
# Decay
# ===============================

def decay_memories():

    alive_m = []
    alive_v = []

    for m,v in zip(memories,vectors):

        m["confidence"] *= 0.995

        if m["confidence"] > 0.2:
            alive_m.append(m)
            alive_v.append(v)
            update_memory(m["id"], m["confidence"], m["last_used"])
        else:
            delete_memory(m["id"])

    memories[:] = alive_m
    vectors[:] = alive_v

    index.reset()
    if vectors:
        index.add(np.vstack(vectors))
>>>>>>> ddf6092 (Initial MemoryFlow submission)
