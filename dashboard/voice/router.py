"""
dashboard/voice/router.py
--------------------------
Audio transcription via ElevenLabs Scribe STT (with Gemini 2.5 Flash fallback).

Purpose:
  Accepts a recorded audio blob (or a plain-text fallback) from the patient
  dashboard's voice-triage widget, transcribes it via ElevenLabs Scribe STT,
  generates a concise clinical triage summary, and returns { transcript, summary }.

Endpoint:
  POST /voice/transcribe
  Content-Type: multipart/form-data

  Fields:
    audio_file : UploadFile  — audio blob (webm / mp4 / ogg / wav / mpeg)
                                OR omit to use text-fallback mode
    text       : str         — symptom text (used when audio_file is absent)

  Auth: Bearer JWT — optional (supports both authenticated patients and guest sessions).

  Returns:
    { "transcript": "...", "summary": "..." }
"""

import base64
import json
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from auth.dependencies import get_current_user, get_optional_current_user
from auth.models import User

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB limit

ALLOWED_AUDIO_MIME_PREFIXES = ("audio/",)
ALLOWED_AUDIO_TYPES = {
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/mpeg",
    "audio/x-m4a",
    "audio/aac",
}

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_STT_MODEL = "scribe_v1"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

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

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/voice", tags=["Voice Transcription"])


async def _transcribe_elevenlabs(audio_bytes: bytes, filename: str, mime: str, api_key: str) -> str:
    """
    Transcribe audio via ElevenLabs Scribe STT API.
    POST https://api.elevenlabs.io/v1/speech-to-text
    """
    headers = {"xi-api-key": api_key}
    files = {"file": (filename or "recording.webm", audio_bytes, mime or "audio/webm")}
    data = {"model_id": ELEVENLABS_STT_MODEL}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data)

    if resp.status_code != 200:
        logger.error("ElevenLabs Scribe error %s: %s", resp.status_code, resp.text[:300])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ElevenLabs Scribe API error {resp.status_code}: {resp.text[:200]}",
        )

    result = resp.json()
    transcript = result.get("text", "").strip()
    return transcript


async def _generate_clinical_summary(transcript: str) -> str:
    """
    Generate a concise clinical triage summary from transcript text using OpenAI gpt-4o-mini
    (with Gemini fallback).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=openai_key)
            prompt = (
                "You are Arogya AI, an empathetic, expert clinical triage assistant.\n"
                f"A patient described their symptoms: \"{transcript}\"\n\n"
                "Generate a concise, professional medical triage summary including:\n"
                "- Key symptoms identified\n"
                "- Severity assessment (mild / moderate / severe)\n"
                "- Likely conditions or differentials\n"
                "- Actionable recommendations (rest, hydration, OTC care, seek medical attention)\n"
                "- Red-flag symptoms to monitor\n\n"
                "Keep the response structured and under 150 words."
            )
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=250,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
        except Exception as exc:
            logger.warning("OpenAI summary generation failed: %s", exc)

    # Fallback to Gemini if Gemini API key is present
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "You are Arogya AI. Generate a concise medical triage summary including "
                                    f"key symptoms, severity, and recommendations for: \"{transcript}\""
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 250},
            }
            res = await _call_gemini(payload, gemini_key)
            return res.get("summary") or res.get("text", "")
        except Exception as exc:
            logger.warning("Gemini summary fallback failed: %s", exc)

    return f"Patient reported: {transcript}. Routine clinical follow-up recommended."


async def _call_gemini(payload: dict, api_key: str) -> dict:
    """POST to Gemini generateContent and return the parsed JSON result dict."""
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
        return {"transcript": "", "summary": raw_text}


@router.post(
    "/transcribe",
    summary="Transcribe patient audio or text and return a clinical triage summary",
    status_code=status.HTTP_200_OK,
)
async def transcribe(
    audio_file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Accepts either:
      - `audio_file` (multipart) — audio blob recorded by the browser
      - `text` (form field)      — plain-text symptom description (fallback)

    At least one must be provided. Audio takes priority when both are sent.
    Uses ElevenLabs Scribe STT when ELEVENLABS_API_KEY is configured,
    falling back to Gemini 2.5 Flash if needed.
    Returns { transcript, summary }.
    """
    # ── Audio path ────────────────────────────────────────────────────────────
    if audio_file is not None:
        mime = (audio_file.content_type or "").split(";")[0].strip().lower()
        if not (mime in ALLOWED_AUDIO_TYPES or mime.startswith("audio/")):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio type '{mime}'. Accepted: webm, mp4, ogg, wav, mpeg.",
            )

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

        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        # 1. Primary: ElevenLabs Scribe STT
        if elevenlabs_key:
            try:
                transcript = await _transcribe_elevenlabs(
                    audio_bytes=audio_bytes,
                    filename=audio_file.filename or "recording.webm",
                    mime=mime,
                    api_key=elevenlabs_key,
                )
                if transcript:
                    summary = await _generate_clinical_summary(transcript)
                    return {
                        "transcript": transcript,
                        "summary": summary,
                    }
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("ElevenLabs transcription failed, checking Gemini fallback: %s", exc)
                if not gemini_key:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"ElevenLabs transcription failed: {exc}",
                    )

        # 2. Fallback: Gemini 2.5 Flash STT
        if gemini_key:
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
            result = await _call_gemini(gemini_payload, gemini_key)
            return {
                "transcript": result.get("transcript", "Transcript unavailable"),
                "summary": result.get("summary", ""),
            }

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Neither ELEVENLABS_API_KEY nor GEMINI_API_KEY is configured on the server.",
        )

    # ── Text fallback path ────────────────────────────────────────────────────
    if text and text.strip():
        safe_text = text.strip()
        summary = await _generate_clinical_summary(safe_text)
        return {
            "transcript": safe_text,
            "summary": summary,
        }

    # ── Neither provided ──────────────────────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Provide either 'audio_file' (audio upload) or 'text' (symptom description).",
    )
