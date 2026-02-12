<<<<<<< HEAD
from transformers import pipeline

clf = pipeline("text-classification", model="./classifier")

print(clf("i am darshan"))
print(clf("i prefer hindi"))
print(clf("ring me at 5pm"))
=======
from transformers import pipeline

clf = pipeline("text-classification", model="./classifier")

print(clf("i am darshan"))
print(clf("i prefer hindi"))
print(clf("ring me at 5pm"))
>>>>>>> ddf6092 (Initial MemoryFlow submission)
