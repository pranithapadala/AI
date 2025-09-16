from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
pairs = [
    ("I love this course.", "This bootcamp is amazing."),
    ("I love this course.", "This setup is so confusing."),
    ("The sky is clear today.", "It is Monday today.")
]
embs = model.encode([s for pair in pairs for s in pair], normalize_embeddings=True)
def cos(a, b): return float(util.cos_sim(a, b))

for i, (a, b) in enumerate(pairs):
    ea = embs[2*i]
    eb = embs[2*i+1]
    print(f"\nPair {i+1}:")
    print("A:", a)
    print("B:", b)
    print("Cosine similarity:", round(cos(ea, eb), 3))

from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"  
)
print(clf(["I love this!", "Meh, it's okay.", "This is awful."]))
