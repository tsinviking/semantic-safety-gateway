import re
from dataclasses import dataclass
from typing import Optional

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s*:\s*override",
    r"you\s+are\s+now\s+dan",
    r"jailbreak",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
}

BANNED_OUTPUT_PATTERNS = [
    r"(?i)(DROP|DELETE|TRUNCATE)\s+TABLE",
    r"(?i)rm\s+-rf\s+/",
]

@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: Optional[str] = None
    modified_text: Optional[str] = None

def validate_input(text: str) -> GuardrailResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(passed=False, blocked_reason="")
    
    modified=text
    pii_found = False

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, modified):
            pii_found = True
            modified = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", modified)
    
    return GuardrailResult(passed=True, modified_text=modified if pii_found else None)

def validate_output(text: str) -> GuardrailResult:
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern,text):
            return GuardrailResult(passed=False, blocked_reason="Destructive command sequence found in generation.")
        return GuardrailResult(passed=True)
