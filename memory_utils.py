import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from db import insert_memory, load_memories, update_memory, delete_memory

# ===============================
# Labels & Constants
# ===============================

LABEL_MAP = {
    "LABEL_0": "preference",
    "LABEL_1": "constraint",
    "LABEL_2": "fact",
    "LABEL_3": "instruction",
    "LABEL_4": "ignore",
    "LABEL_5": "command"
}

PROFILE_TYPES = ["fact", "preference", "constraint", "instruction", "command"]

# ===============================
# Models
# ===============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = pipeline("text-classification", model="./classifier", top_k=1)

# ===============================
# FAISS Setup
# ===============================

DIM = 384
index = faiss.IndexFlatIP(DIM)

# ===============================
# Memory State
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
    if len(t.split()) < 3: return True
    if re.search(r"[a-z]{12,}", t): return True
    return False

def is_identity_query(q):
    return any(x in q.lower() for x in ["about me", "who am i", "my info"])

# ===============================
# Core Logic: Semantic Chunking
# ===============================

def extract_semantic_chunks(text):
    """
    Splits text into logical units based on grammar rules.
    Handles: Contrast (but/however), Conditions (unless/if), and Exceptions (except).
    """
    chunks = []
    
    # 1. Split by major punctuation first (. ? ! \n ;)
    # We keep the delimiter to know where sentences end
    major_parts = re.split(r"([.?!;\n])", text)
    
    # Re-assemble roughly to avoid dangling punctuation
    sentences = []
    current_sent = ""
    for p in major_parts:
        if p in [".", "?", "!", ";", "\n"]:
            current_sent += p
            sentences.append(current_sent.strip())
            current_sent = ""
        else:
            current_sent += p
    if current_sent: sentences.append(current_sent.strip())

    # 2. Sub-split complex sentences using Discourse Markers
    # These words signal a shift in logic (e.g., from Preference -> Constraint)
    markers = [
        "but", "however", "yet", "although", "though", 
        "except", "unless", "otherwise", "save for", "apart from"
    ]
    
    # Regex pattern: looks for ", keyword" or "; keyword" or " keyword "
    # We capture the marker to keep it with the *second* half (context preservation)
    pattern = r"(?i)(,|\.|;)?\s*\b(" + "|".join(markers) + r")\b"

    for sent in sentences:
        if not sent: continue
        
        # Split on the markers
        parts = re.split(pattern, sent)
        
        # The split returns: [Part1, Punctuation, Marker, Part2, Punctuation, Marker, Part3...]
        # We need to reconstruct this intelligently.
        
        current_chunk = parts[0].strip()
        
        # Iterate over the rest of the split result in steps of 3 (Punct, Marker, Text)
        i = 1
        while i < len(parts):
            marker = parts[i+1] if (i+1) < len(parts) else ""
            content = parts[i+2] if (i+2) < len(parts) else ""
            
            # If the chunk is substantial, save the previous one and start a new one
            if len(content.split()) > 2: # heuristic: fragments aren't constraints
                if current_chunk: chunks.append(current_chunk)
                # Start new chunk WITH the marker (e.g. "but never use...")
                current_chunk = f"{marker} {content}".strip()
            else:
                # If it's a tiny fragment, just append it to current
                current_chunk += f" {marker} {content}"
            
            i += 3
            
        if current_chunk:
            chunks.append(current_chunk)

    return chunks

# ===============================
# Load & Retrieve (Standard)
# ===============================

def load_existing():
    global memories, vectors, profile, index
    rows = load_memories()
    if not rows: return
    memories = []
    vectors = []
    profile = []
    texts = []
    for r in rows:
        mem = {"id": r[0], "content": r[1], "type": r[2], "confidence": r[3], "turn": r[4], "last_used": r[5]}
        memories.append(mem)
        if mem["type"] in PROFILE_TYPES: profile.append(mem)
        texts.append(mem["content"])
    if texts:
        vecs = embed(texts)
        vectors.extend(vecs)
        index.add(vecs)

load_existing()

def detect_type(text):
    prediction = classifier(text)
    if isinstance(prediction, list) and isinstance(prediction[0], list): result = prediction[0][0]
    elif isinstance(prediction, list): result = prediction[0]
    else: result = prediction
    return LABEL_MAP[result["label"]]

# ===============================
# Add Memory (Generalized)
# ===============================

def add_memory(text, current_turn):
    print(f"\n--- Processing: '{text}' ---")
    
    # 1. Use the new smart splitter
    final_parts = extract_semantic_chunks(text)
    print(f"DEBUG: Chunks -> {final_parts}")

    for p in final_parts:
        if is_noise(p): continue
        if any(m["content"].lower() == p.lower() for m in memories): continue

        # 2. Classification + Keyword Safety Net
        # We still keep the keyword override because it's the safest way to ensure
        # the 'Constraint' label is applied for the hackathon criteria.
        lower_p = p.lower()
        if any(w in lower_p for w in ["never", "do not", "don't", "must", "avoid", "ensure"]):
            t = "constraint"
        else:
            t = detect_type(p)

        if t == "ignore": continue

        # 3. Semantic De-duplication (Hard Replace)
        updated = False
        if len(memories) > 0:
            q_vec = embed([p])
            D, I = index.search(q_vec, 1)
            
            if D[0][0] > 0.90:
                idx = I[0][0]
                existing_mem = memories[idx]
                
                # Replace if new one is significantly longer/better
                if len(p) > len(existing_mem["content"]) + 5:
                    print(f"DEBUG: Replacing ID {existing_mem['id']}")
                    delete_memory(existing_mem["id"])
                    row_id = insert_memory(p, t, 1.0, current_turn, current_turn)
                    memories[idx] = {
                        "id": row_id, "content": p, "type": t,
                        "confidence": 1.0, "turn": current_turn, "last_used": current_turn
                    }
                    index.add(np.array([q_vec[0]]))
                    updated = True
                else:
                    existing_mem["last_used"] = current_turn
                    update_memory(existing_mem["id"], existing_mem["confidence"], current_turn)
                    updated = True
        
        if updated: continue

        # 4. Insert
        vec = embed([p])[0]
        # Constraints get max confidence immediately
        conf = 1.0 if t in ["constraint", "instruction"] else 0.9
        
        row_id = insert_memory(p, t, conf, current_turn, current_turn)
        
        mem = {
            "id": row_id, "content": p, "type": t,
            "confidence": conf, "turn": current_turn, "last_used": current_turn
        }
        memories.append(mem)
        vectors.append(vec)
        index.add(np.array([vec]))
        if t in PROFILE_TYPES: profile.append(mem)

def retrieve_memories(query, k=5):
    if is_identity_query(query): return profile[:k]
    if not memories: return []

    q = embed([query])
    D, I = index.search(q, k*2)
    results = []
    seen_ids = set()

    for sim, idx in zip(D[0], I[0]):
        if sim < 0.60: continue
        if idx >= len(memories): continue
        m = memories[idx]
        if m["id"] in seen_ids: continue
        
        m["confidence"] = min(m["confidence"] + 0.05, 1.0)
        m["last_used"] = turn
        update_memory(m["id"], m["confidence"], m["last_used"])
        results.append(m)
        seen_ids.add(m["id"])
        if len(results) >= k: break
    return results

def summarize_memories(memories):
    if not memories: return "No memories yet."
    return "; ".join([f"[{m['type']}] {m['content']}" for m in memories])

def decay_memories():
    pass