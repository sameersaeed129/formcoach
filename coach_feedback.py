"""
coach_feedback.py
---------------------
Turns the numeric outputs of the pipeline (form classification, DTW
similarity score, joint angle stats) into a human-readable coaching note.

Two modes:
  - Template mode (default, always available, zero setup): fills a
    hand-written, biomechanically-accurate template per form-error class.
  - LLM mode (optional): sends the same structured facts to a local Ollama
    model to generate more natural, varied phrasing. Falls back to
    template mode automatically if Ollama isn't running, so the app never
    breaks because of a missing LLM.
"""

from dataclasses import dataclass
from typing import Optional
import json
import urllib.request
import urllib.error

TEMPLATES = {
    "good_form": (
        "Rep {rep_number}: Solid form. Knee angle reached {min_knee:.0f}\u00b0 at the bottom "
        "(good depth) and your torso lean stayed controlled. Similarity to reference movement: "
        "{similarity:.0f}/100. Keep this pattern consistent across all your sets."
    ),
    "knee_valgus": (
        "Rep {rep_number}: Your knees moved inward (valgus collapse) during the descent -- "
        "a common pattern that increases stress on the ACL and MCL. Similarity to reference "
        "movement: {similarity:.0f}/100. Try actively pushing your knees outward, tracking "
        "over your toes, and consider adding banded lateral walks to strengthen your hip "
        "abductors."
    ),
    "insufficient_depth": (
        "Rep {rep_number}: You didn't reach full depth -- knee angle only reached {min_knee:.0f}\u00b0 "
        "(a full squat typically reaches ~90\u00b0 or below). Similarity to reference movement: "
        "{similarity:.0f}/100. Work on ankle and hip mobility, and consider box squats to build "
        "confidence sitting into a deeper range of motion."
    ),
    "back_rounding": (
        "Rep {rep_number}: Your torso leaned forward more than {torso_lean:.0f}\u00b0, suggesting your "
        "lower back may be rounding under load instead of your hips doing the work. Similarity "
        "to reference movement: {similarity:.0f}/100. Focus on bracing your core before descending "
        "and keeping your chest tall throughout the movement."
    ),
}


@dataclass
class RepFeedback:
    rep_number: int
    predicted_label: str
    confidence: float
    similarity_score: float
    text: str


def _template_feedback(rep_number: int, label: str, confidence: float, similarity: float,
                        min_knee: float, torso_lean: float) -> str:
    template = TEMPLATES.get(label, TEMPLATES["good_form"])
    return template.format(
        rep_number=rep_number, similarity=similarity,
        min_knee=min_knee, torso_lean=torso_lean,
    )


def _try_ollama_feedback(rep_number: int, label: str, confidence: float, similarity: float,
                          min_knee: float, torso_lean: float,
                          model: str = "llama3.2", host: str = "http://localhost:11434",
                          timeout: float = 4.0) -> Optional[str]:
    """Attempts to generate feedback via a local Ollama server. Returns None
    (never raises) if Ollama isn't reachable, so callers can transparently
    fall back to the template."""
    prompt = (
        f"You are a concise, encouraging strength coach. A lifter just performed a squat rep "
        f"with this data: form classification = '{label}' (model confidence {confidence:.0%}), "
        f"movement similarity to ideal reference = {similarity:.0f}/100, minimum knee angle = "
        f"{min_knee:.0f} degrees, peak torso forward lean = {torso_lean:.0f} degrees. "
        f"Write ONE short paragraph (2-3 sentences) of specific, actionable coaching feedback "
        f"for this single rep. Be direct and technical, not generic."
    )
    try:
        payload = json.dumps({
            "model": model, "prompt": prompt, "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def generate_feedback(rep_number: int, label: str, confidence: float, similarity: float,
                       min_knee: float, torso_lean: float,
                       use_llm: bool = False) -> RepFeedback:
    text = None
    if use_llm:
        text = _try_ollama_feedback(rep_number, label, confidence, similarity, min_knee, torso_lean)
    if text is None:
        text = _template_feedback(rep_number, label, confidence, similarity, min_knee, torso_lean)

    return RepFeedback(
        rep_number=rep_number, predicted_label=label, confidence=confidence,
        similarity_score=similarity, text=text,
    )
