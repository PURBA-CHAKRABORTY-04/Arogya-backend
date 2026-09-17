"""
dashboard/voice/router.py
--------------------------
Path B — Server-side audio transcription via Google Gemini 2.5 Flash.

Purpose:
  Accepts a recorded audio blob (or a plain-text fallback) from the patient
  dashboard's voice-triage widget, calls the Gemini API server-side, and
  returns { transcript, summary }.

Kept entirely separate from Path A (symptom_triage/router.py / OpenAI
GPT-4o-mini), which handles the multi-turn clinical triage chatbot.

Endpoint:
  POST /voice/transcribe
  Content-Type: multipart/form-data

  Fields:
    audio_file : UploadFile  — audio blob (webm / mp4 / ogg / wav / mpeg)
                               OR omit to use text-fallback mode
    text       : str         — symptom text (used when audio_file is absent)

  Auth: Bearer JWT — requires a valid patient session.

  Returns:
    { "transcript": "...", "summary": "..." }
"""

import base64
import json
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from auth.dependencies import get_current_user
from auth.models import User

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB — Gemini inline_data limit

ALLOWED_AUDIO_MIME_PREFIXES = ("audio/",)          # webm, mp4, ogg, wav, mpeg …
ALLOWED_AUDIO_TYPES = {                            # explicit allowlist for safety
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/mpeg",
    "audio/x-m4a",
    "audio/aac",
}

GEMINI_MODEL  = "gemini-2.5-flash"
GEMINI_BASE   = "https://generativelanguage.googleapis.com/v1beta/models"

# ── Prompt shared by both audio and text paths ────────────────────────────────
AUDIO_SYSTEM_PROMPT = """\
You are Arogya AI, an expert clinical triage assistant.
A patient has described their symptoms.

Your tasks:
1. Transcribe exactly what the patient said (verbatim, first-person).
2. Generate a concise, professional medical triage summary including:
   - Key symptoms identified
   - Severity assessment (mild / moderate / severe)
   - Likely condition(s) or differentials
   - Actionable recommendations (rest, hydration, OTC meds, seek care, etc.)
   - Any red-flag symptoms to watch for

Return ONLY a valid JSON object (no markdown, no code fences), exactly like this:
{
  "transcript": "exact words the patient spoke or typed",
  "summary": "professional triage summary paragraph"
}"""

TEXT_PROMPT_TEMPLATE = """\
You are Arogya AI, an expert clinical triage assistant.
A patient described their symptoms in text: "{text}"

Generate a concise medical triage summary including:
- Key symptoms identified
- Severity assessment
- Recommendations

Return ONLY a valid JSON object (no markdown, no code fences):
{{
  "transcript": "{text_escaped}",
  "summary": "your summary here"
}}"""

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/voice", tags=["Voice Transcription"])


def _get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured on the server.",
        )
    return key


async def _call_gemini(payload: dict, api_key: str) -> dict:
    """
    POST to Gemini generateContent and return the parsed JSON result dict.
    Raises HTTPException on Gemini errors.
    """
    url = f"{GEMINI_BASE}/{GEMINI_MODEL}:generateContent?key={api_key}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API error {resp.status_code}: {resp.text[:300]}",
        )

    data = resp.json()
    raw_text: str = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        # Gemini returned non-JSON — surface the raw text as the summary
        return {"transcript": "", "summary": raw_text}


@router.post(
    "/transcribe",
    summary="Transcribe patient audio or text and return a clinical triage summary",
    status_code=status.HTTP_200_OK,
)
async def transcribe(
    audio_file: Optional[UploadFile] = File(default=None),
    text:       Optional[str]        = Form(default=None),
    current_user: User               = Depends(get_current_user),
):
    """
    Accepts either:
      - `audio_file` (multipart) — audio blob recorded by the browser
      - `text` (form field)      — plain-text symptom description (fallback)

    At least one must be provided. Audio takes priority when both are sent.
    Returns { transcript, summary }.
    """
    api_key = _get_gemini_key()

    # ── Audio path ────────────────────────────────────────────────────────────
    if audio_file is not None:
        # Validate MIME type
        mime = (audio_file.content_type or "").split(";")[0].strip().lower()
        if not (mime in ALLOWED_AUDIO_TYPES or mime.startswith("audio/")):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio type '{mime}'. Accepted: webm, mp4, ogg, wav, mpeg.",
            )

        # Read + validate size
        audio_bytes = await audio_file.read()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Audio file exceeds the 25 MB limit.",
            )
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty.",
            )

        # Base64-encode for Gemini inline_data
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        gemini_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime or "audio/webm",
                                "data": b64_audio,
                            }
                        },
                        {"text": AUDIO_SYSTEM_PROMPT},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }

        result = await _call_gemini(gemini_payload, api_key)
        return {
            "transcript": result.get("transcript", "Transcript unavailable"),
            "summary":    result.get("summary", ""),
        }

    # ── Text fallback path ────────────────────────────────────────────────────
    if text and text.strip():
        safe_text = text.strip()
        text_escaped = safe_text.replace('"', "'")

        prompt = (
            f'You are Arogya AI, an expert clinical triage assistant.\n'
            f'A patient described their symptoms: "{safe_text}"\n\n'
            f'Generate a concise medical triage summary including:\n'
            f'- Key symptoms identified\n'
            f'- Severity assessment (mild / moderate / severe)\n'
            f'- Recommendations (rest, OTC meds, seek care, etc.)\n\n'
            f'Return ONLY a valid JSON object (no markdown, no code fences):\n'
            f'{{"transcript": "{text_escaped}", "summary": "your summary here"}}'
        )

        gemini_payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
            },
        }

        result = await _call_gemini(gemini_payload, api_key)
        return {
            "transcript": result.get("transcript", safe_text),
            "summary":    result.get("summary", ""),
        }

    # ── Neither provided ──────────────────────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Provide either 'audio_file' (audio upload) or 'text' (symptom description).",
    )
