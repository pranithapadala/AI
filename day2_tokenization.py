from transformers import AutoTokenizer
import torch
model_id = "distilbert-base-uncased"
tok= AutoTokenizer.from_pretrained(model_id)

sentences = [
    "I love this place!!",
    "It is Monday today.",
    "Learning this is fun"
]

enc = tok(sentences, padding=True, truncation= True,return_tensors="pt")

print("Input IDs shape:", enc["input_ids"].shape)         
print("Attention mask shape:", enc["attention_mask"].shape)


print("\nBatch decode (skip special tokens):")
print(tok.batch_decode(enc["input_ids"], skip_special_tokens=True))

print("\nTokens per sentence (including special tokens):")
for i, ids in enumerate(enc["input_ids"]):
    ids_list = ids.tolist()  
    tokens = tok.convert_ids_to_tokens(ids_list)
    print(f"\nSentence {i+1}: {sentences[i]}")
    print(tokens)

print("\nInput IDs (first row):", enc["input_ids"][0].tolist())
print("Attention mask (first row):", enc["attention_mask"][0].tolist())