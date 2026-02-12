<<<<<<< HEAD
from memory import add_memory, retrieve_memories, next_turn
import random
import string

LOG_FILE = "stress_results.txt"

facts = [
    "My name is Darshan",
    "I prefer Hindi",
    "I live in Mumbai",
    "My favorite tech is AI",
]

print("Adding base memories...")

for f in facts:
    add_memory(f, next_turn())

print("Injecting 1000 noise memories...")

for i in range(1000):
    junk = ''.join(random.choices(string.ascii_lowercase, k=20))
    add_memory(f"I saw {junk}", next_turn())

print("Running retrieval...")

query = "what do you know about me"

results = retrieve_memories(query, k=5)

with open(LOG_FILE, "w") as f:
    f.write("QUERY:\n")
    f.write(query + "\n\n")

    f.write("RESULTS:\n")

    for r in results:
        f.write(str(r) + "\n")

print("\nRetrieved:")
for r in results:
    print(r)

print("\nSaved to stress_results.txt")
=======
from memory import add_memory, retrieve_memories, next_turn
import random
import string

LOG_FILE = "stress_results.txt"

facts = [
    "My name is Darshan",
    "I prefer Hindi",
    "I live in Mumbai",
    "My favorite tech is AI",
]

print("Adding base memories...")

for f in facts:
    add_memory(f, next_turn())

print("Injecting 1000 noise memories...")

for i in range(1000):
    junk = ''.join(random.choices(string.ascii_lowercase, k=20))
    add_memory(f"I saw {junk}", next_turn())

print("Running retrieval...")

query = "what do you know about me"

results = retrieve_memories(query, k=5)

with open(LOG_FILE, "w") as f:
    f.write("QUERY:\n")
    f.write(query + "\n\n")

    f.write("RESULTS:\n")

    for r in results:
        f.write(str(r) + "\n")

print("\nRetrieved:")
for r in results:
    print(r)

print("\nSaved to stress_results.txt")
>>>>>>> ddf6092 (Initial MemoryFlow submission)
