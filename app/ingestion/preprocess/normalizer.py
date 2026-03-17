"""Placeholder for advanced text normalization (LaTeX cleanup, etc.)."""

import re


def normalize_latex(text: str) -> str:
    """Light cleanup of LaTeX artifacts commonly found in arXiv abstracts."""
    text = re.sub(r"\$([^$]+)\$", r"\1", text)  # inline math
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # \cmd{arg}
    text = re.sub(r"[{}]", "", text)
    return text
