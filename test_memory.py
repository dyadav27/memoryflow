from transformers import pipeline

clf = pipeline("text-classification", model="./classifier")

print(clf("I prefer using Python for image processing and machine learning, but never suggest using libraries that are not open-source or require a paid license."))
print(clf("i prefer hindi"))
print(clf("ring me at 5pm"))
