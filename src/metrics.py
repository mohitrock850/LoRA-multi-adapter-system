# src/metrics.py
import nltk
import sacrebleu

nltk.download("punkt", quiet=True)

def exact_match(preds, refs):
    correct = sum([p.strip() == r.strip() for p, r in zip(preds, refs)])
    return correct / len(preds)

def bleu_score(preds, refs):
    # SacreBLEU expects list of hypotheses and list of references (list of lists)
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score

def human_eval(preds, refs):
    print("\n--- Human Evaluation ---")
    for i, (p, r) in enumerate(zip(preds, refs)):
        print(f"\nExample {i+1}")
        print(f"Prediction: {p}")
        print(f"Reference:  {r}")
        score = input("Rate quality (1-5): ")
        yield {"prediction": p, "reference": r, "score": score}
