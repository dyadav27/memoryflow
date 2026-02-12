<<<<<<< HEAD
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")

def init():

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            type TEXT,
            confidence REAL,
            turn INTEGER,
            last_used INTEGER
        )
        """)

init()

# ===============================
# Insert
# ===============================

def insert_memory(content, mtype, confidence, turn, last_used):

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO memories (content, type, confidence, turn, last_used)
    VALUES (?, ?, ?, ?, ?)
    """, (content, mtype, confidence, turn, last_used))

    conn.commit()

    return cur.lastrowid   # <<< CRITICAL


# ===============================
# Load
# ===============================

def load_memories():

    cur = conn.cursor()
    cur.execute("SELECT * FROM memories")
    return cur.fetchall()

# ===============================
# Update
# ===============================

def update_memory(mid, confidence, last_used):

    if mid is None:
        return

    cur = conn.cursor()

    cur.execute("""
    UPDATE memories
    SET confidence=?, last_used=?
    WHERE id=?
    """, (confidence, last_used, mid))

    conn.commit()

# ===============================
# Delete
# ===============================

def delete_memory(mid):

    if mid is None:
        return

    cur = conn.cursor()
    cur.execute("DELETE FROM memories WHERE id=?", (mid,))
    conn.commit()
=======
import sqlite3

conn = sqlite3.connect("memory.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL")

def init():

    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            type TEXT,
            confidence REAL,
            turn INTEGER,
            last_used INTEGER
        )
        """)

init()

# ===============================
# Insert
# ===============================

def insert_memory(content, mtype, confidence, turn, last_used):

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO memories (content, type, confidence, turn, last_used)
    VALUES (?, ?, ?, ?, ?)
    """, (content, mtype, confidence, turn, last_used))

    conn.commit()

    return cur.lastrowid   # <<< CRITICAL


# ===============================
# Load
# ===============================

def load_memories():

    cur = conn.cursor()
    cur.execute("SELECT * FROM memories")
    return cur.fetchall()

# ===============================
# Update
# ===============================

def update_memory(mid, confidence, last_used):

    if mid is None:
        return

    cur = conn.cursor()

    cur.execute("""
    UPDATE memories
    SET confidence=?, last_used=?
    WHERE id=?
    """, (confidence, last_used, mid))

    conn.commit()

# ===============================
# Delete
# ===============================

def delete_memory(mid):

    if mid is None:
        return

    cur = conn.cursor()
    cur.execute("DELETE FROM memories WHERE id=?", (mid,))
    conn.commit()
>>>>>>> ddf6092 (Initial MemoryFlow submission)
