from transformers import AutoTokenizer, AutoModel
import torch

MODEL = "bert-base-uncased"
SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}
PUNCT   = {".", ",", "!", "?", ";", ":"}

def topk_content(attn_row, tokens, k=3):
    # mask out specials/punct for ranking (keep self allowed)
    masked = attn_row.clone()
    for j, tok in enumerate(tokens):
        if tok in SPECIAL or tok in PUNCT:
            masked[j] = float("-inf")
    # if everything got masked (e.g., very short seq), fall back to original
    if torch.isinf(masked).all():
        masked = attn_row
    idx = torch.topk(masked, min(k, masked.shape[0]))[1].tolist()
    return idx

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, output_attentions=True).eval()

sentence = "The cat sat on the mat because the mat was dirty."
enc = tok(sentence, return_tensors="pt")

with torch.no_grad():
    out = model(**enc)

# out.attentions: tuple of length = num_layers, each [batch, heads, seq, seq]
layers = torch.stack([a[0] for a in out.attentions], dim=0)  # [layers, heads, seq, seq]

# average over layers & heads -> [seq, seq]
attn_avg = layers.mean(dim=(0, 1))

tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])

print("Tokens:", tokens, "\n")
for i, t in enumerate(tokens):
    idxs = topk_content(attn_avg[i], tokens, k=3)
    items = [f"{tokens[j]}({attn_avg[i, j].item():.3f})" for j in idxs]
    print(f"{i:02d} {t:>8s} → " + ", ".join(items))
