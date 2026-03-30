import re
import logging
from dataclasses import dataclass, field
from config import config

log = logging.getLogger("dlp.tokenizer")

# Cloud DLP infoTypes to inspect and tokenize
_DLP_INFO_TYPES = [
    {"name": "PERSON_NAME"},
    {"name": "EMAIL_ADDRESS"},
    {"name": "PHONE_NUMBER"},
    {"name": "US_SOCIAL_SECURITY_NUMBER"},
    {"name": "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"},
    {"name": "CREDIT_CARD_NUMBER"},
    {"name": "STREET_ADDRESS"},
]

# Maximum characters Cloud DLP accepts per request
_DLP_MAX_CHARS = 524288


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


def _tokenize_with_dlp(text, ctx):
    """
    Tokenize PII using the Cloud DLP inspect API.
    Inspects text for configured infoTypes, replaces each finding with a
    reversible surrogate token, and records the mapping in ctx.
    Falls back to local regex on API failure.
    """
    try:
        from google.cloud import dlp_v2

        client = dlp_v2.DlpServiceClient()
        parent = f"projects/{config.PROJECT_ID}/locations/global"

        item = {"value": text[:_DLP_MAX_CHARS]}
        inspect_config = {
            "info_types": _DLP_INFO_TYPES,
            "include_quote": True,
            "min_likelihood": dlp_v2.Likelihood.LIKELY,
        }

        response = client.inspect_content(
            request={
                "parent": parent,
                "inspect_config": inspect_config,
                "item": item,
            }
        )

        findings = response.result.findings
        if not findings:
            return text

        # Sort findings by byte offset descending so replacements don't
        # shift the positions of earlier findings in the string
        findings_sorted = sorted(
            findings,
            key=lambda f: f.location.byte_range.start,
            reverse=True,
        )

        text_bytes = text.encode("utf-8")
        for finding in findings_sorted:
            info_type = finding.info_type.name
            quote = finding.quote
            if not quote:
                continue
            token = ctx.register(quote, info_type)
            start = finding.location.byte_range.start
            end = finding.location.byte_range.end
            text_bytes = text_bytes[:start] + token.encode("utf-8") + text_bytes[end:]

        return text_bytes.decode("utf-8")

    except Exception as e:
        log.warning("Cloud DLP tokenization failed, falling back to local regex: %s", e)
        return _tokenize_local_fallback(text, ctx)


def _tokenize_local_fallback(text, ctx):
    """
    Local regex fallback covering the most common PII patterns.
    Used when the Cloud DLP API is unavailable.
    """
    patterns = [
        ("EMAIL_ADDRESS", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        ("PHONE_NUMBER",  r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b"),
        ("US_SOCIAL_SECURITY_NUMBER", r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    ]
    for info_type, pattern in patterns:
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
    return _tokenize_with_dlp(text, ctx)


def detokenize(text, ctx=None):
    if not text:
        return text
    return (ctx or TokenizationContext()).detokenize(text)
