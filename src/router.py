# src/router.py
import re
from dataclasses import dataclass

@dataclass
class Route:
    name: str
    confidence: float

class KeywordRouter:
    def __init__(self, mapping=None):
        # mapping domain -> list of keywords
        self.mapping = mapping or {
            "law": ["law", "contract", "tort", "plaintiff", "defendant", "statute", "court"],
            "med": ["symptom", "diagnosis", "treatment", "drug", "disease", "patient", "clinic"],
            "code": ["python", "function", "error", "compile", "bug", "javascript", "api"]
        }
        self._compile()

    def _compile(self):
        self.patterns = {dom: [re.compile(r"\b" + re.escape(k) + r"\b", re.I) for k in kwlist] for dom, kwlist in self.mapping.items()}

    def route(self, text):
        scores = {d: sum(1 for p in pats if p.search(text)) for d, pats in self.patterns.items()}
        winner = max(scores, key=scores.get)
        conf = scores[winner] / (sum(scores.values()) + 1e-9)
        if scores[winner] == 0:
            return Route("general", 0.0)
        return Route(winner, conf)
