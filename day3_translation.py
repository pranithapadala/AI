from transformers import pipeline
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
sentences = [
    "I love learning about artificial intelligence.",
    "Transformers are changing the world of NLP.",
    "The weather is nice today."
]
for s in sentences:
    result = translator(s)[0]['translation_text']
    print(f"EN: {s}")
    print(f"FR: {result}\n")