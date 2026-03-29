import re
import logging
from dataclasses import dataclass, field
from config import config

log = logging.getLogger("dlp.tokenizer")


@dataclass
class TokenizationContext:
    _original_to_token: dict = field(default_factory=dict)
    _token_to_original: dict = field(default_factory=dict)
    _counters:          dict = field(default_factory=dict)

    def _next_token(self, info_type):
        n = self._counters.get(info_type, 0) + 1
        self._counters[info_type] = n
        return f"[{info_type}_{n}]"

    def register(self, original, info_type):
        if original in self._original_to_token:
            return self._original_to_token[original]
        token = self._next_token(info_type)
        self._original_to_token[original] = token
        self._token_to_original[token] = original
        return token

    def detokenize(self, text):
        for token, original in self._token_to_original.items():
            text = text.replace(token, original)
        return text


_PATTERNS = [
    ("EMAIL_ADDRESS", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("PHONE_NUMBER",  r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b"),
    ("US_SSN",        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
]


def _tokenize_local(text, ctx):
    for info_type, pattern in _PATTERNS:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            token = ctx.register(match.group(), info_type)
            text = text[:match.start()] + token + text[match.end():]
    return text


def new_context():
    return TokenizationContext()

def tokenize(text, ctx=None):
    if not text:
        return text
    ctx = ctx or TokenizationContext()
    return _tokenize_local(text, ctx)

def detokenize(text, ctx=None):
    if not text:
        return text
    return (ctx or TokenizationContext()).detokenize(text)
