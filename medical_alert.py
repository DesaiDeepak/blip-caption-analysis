"""
Medical Alerting Layer
──────────────────────
Lightweight keyword-based analysis of BLIP-generated captions.
No external dependencies — uses only Python builtins.

Usage:
    from medical_alert import detect_medical_issue
    status, explanation, confidence = detect_medical_issue(caption)
"""

# ── keyword dictionaries (lowercase) ─────────────────────────────────
CRITICAL_KEYWORDS = [
    "severe", "collapse", "bleeding", "hemorrhage", "rupture",
    "pneumothorax", "edema", "embolism", "obstruction", "abscess",
    "necrosis", "perforation", "sepsis",
]

WARNING_KEYWORDS = [
    "opacity", "lesion", "fracture", "tumor", "infection",
    "nodule", "mass", "effusion", "inflammation", "consolidation",
    "cardiomegaly", "atelectasis", "calcification", "thickening",
    "enlarged", "abnormal", "displacement", "narrowing",
]

NORMAL_HINTS = [
    "normal", "clear", "unremarkable", "no abnormality",
    "healthy", "no significant", "within normal",
]


def detect_medical_issue(caption: str):
    """
    Analyse a caption string for medical keywords.

    Returns
    -------
    status : str
        One of ``"CRITICAL"``, ``"WARNING"``, or ``"NORMAL"``.
    explanation : str
        A short, human-readable summary.
    confidence : float
        A 0-1 heuristic score (higher → more keywords matched).
    """
    text = caption.lower()

    # ── collect matched keywords ──────────────────────────────────
    crit_matches = [kw for kw in CRITICAL_KEYWORDS if kw in text]
    warn_matches = [kw for kw in WARNING_KEYWORDS if kw in text]
    norm_matches = [kw for kw in NORMAL_HINTS if kw in text]

    total_medical = len(crit_matches) + len(warn_matches)

    # ── decide status ─────────────────────────────────────────────
    if crit_matches:
        status = "CRITICAL"
        explanation = (
            f"🚨 Critical finding(s) detected: {', '.join(crit_matches)}. "
            "Immediate medical attention recommended."
        )
        confidence = min(0.6 + 0.1 * len(crit_matches), 1.0)

    elif warn_matches:
        status = "WARNING"
        explanation = (
            f"⚠️ Potential issue(s) found: {', '.join(warn_matches)}. "
            "Further review advised."
        )
        confidence = min(0.4 + 0.1 * len(warn_matches), 0.9)

    elif norm_matches:
        status = "NORMAL"
        explanation = "✅ Caption appears to describe a normal finding."
        confidence = min(0.5 + 0.1 * len(norm_matches), 0.95)

    else:
        status = "NORMAL"
        explanation = (
            "ℹ️ No specific medical keywords detected in caption. "
            "Treat as general image."
        )
        confidence = 0.3  # low confidence — caption may not be medical

    return status, explanation, confidence
