from memory import add_memory, retrieve_memories, next_turn
import random
import string

LOG_FILE = "stress_results.txt"

facts = [
    "Translate all technical definitions into Hindi to help me understand the core concepts, but keep the specific programming syntax and keywords in English.",
    "Stop providing long-winded introductions in your responses and immediately list five GitHub repositories for Generative AI projects that do not use OpenAI's API.",
    "I prefer using Python for image processing and machine learning, but never suggest using libraries that are not open-source or require a paid license.",
    " Always use LaTeX for mathematical formulas like $E=mc^2$ but use standard text for simple percentages or temperatures.",
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
