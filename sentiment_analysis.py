from transformers import pipeline
classifier = pipeline("sentiment-analysis")
sentences = [
    "I love this place!!",
    "It is Monday today.",
    "Learning this is fun"
]

for text in sentences:
    result = classifier(text)[0]
    print(f"Text: {text}")
    print(f" -> Label: {result ['label']}, Confidence: {result['score']: .4f}")
    print()