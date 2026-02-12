<<<<<<< HEAD
from transformers import pipeline

clf = pipeline("text-classification", model="./classifier")

tests = [
    "i am darshan",
    "i prefer hindi",
    "remember this",
    "do not save this",
    "ring me at 5pm"
]

for t in tests:
    print(t, clf(t))
=======
from transformers import pipeline

clf = pipeline("text-classification", model="./classifier")

tests = [
    "i am darshan",
    "i prefer hindi",
    "remember this",
    "do not save this",
    "ring me at 5pm"
]

for t in tests:
    print(t, clf(t))
>>>>>>> ddf6092 (Initial MemoryFlow submission)
