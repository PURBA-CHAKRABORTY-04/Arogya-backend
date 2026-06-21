# -*- coding: utf-8 -*-
"""
Diagnostic script: checks each step of the biometric extraction pipeline.
Run from the cognocareai-backend directory:
  python debug_pipeline.py <path_to_pdf>
"""

import io
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

# ── 1. Check table exists ──────────────────────────────────────────────────────
print("\n[1] Checking biometric_metrics table...")
try:
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name='biometric_metrics'"
        ))
        exists = result.scalar()
    if exists:
        print("    OK — table exists")
    else:
        print("    MISSING — table does not exist. Restart FastAPI to create it.")
        sys.exit(1)
except Exception as e:
    print(f"    ERROR — {e}")
    sys.exit(1)

# ── 2. Check row count ─────────────────────────────────────────────────────────
print("\n[2] Checking row count in biometric_metrics...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM biometric_metrics"))
        count = result.scalar()
    print(f"    Total rows: {count}")
except Exception as e:
    print(f"    ERROR — {e}")

# ── 3. Test Unstructured.io API ────────────────────────────────────────────────
print("\n[3] Testing Unstructured.io API...")
UNSTRUCTURED_API_KEY  = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_ENDPOINT = "https://api.unstructuredapp.io/general/v0/general"

pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
if not pdf_path:
    print("    SKIPPED — pass a PDF path as argument to test parsing")
    print("    Usage: python debug_pipeline.py path/to/report.pdf")
else:
    try:
        import httpx
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        resp = httpx.post(
            UNSTRUCTURED_ENDPOINT,
            headers={"unstructured-api-key": UNSTRUCTURED_API_KEY},
            files={"files": ("document.pdf", io.BytesIO(file_bytes), "application/pdf")},
            data={"output_format": "application/json"},
            timeout=60.0,
        )
        print(f"    HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            elements = resp.json()
            text_parts = [e.get("text", "") for e in elements if e.get("text")]
            full_text = "\n".join(text_parts)
            print(f"    Elements returned: {len(elements)}")
            print(f"    Total text chars: {len(full_text)}")
            print(f"    Sample (first 300 chars):\n    {full_text[:300]!r}")
        else:
            print(f"    ERROR response: {resp.text[:500]}")
            sys.exit(1)
    except Exception as e:
        print(f"    EXCEPTION — {e}")
        sys.exit(1)

# ── 4. Test GPT extraction ─────────────────────────────────────────────────────
    print("\n[4] Testing GPT-4o metric extraction...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        GPT_SYSTEM_PROMPT = """You are a clinical data extractor.
Given raw text from a medical document, extract every measurable health metric.
Return ONLY a valid JSON array. Each item:
{"title":"...","subtitle":"...","baseline":"...","current":"...","var_text":"...","var_type":"up|down|neutral","status":"Stable|Alert|Monitoring|Improved"}
Return [] if no metrics found."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Medical document text:\n\n{full_text[:8000]}"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        print(f"    GPT raw response (first 500 chars): {raw[:500]!r}")

        # Strip markdown if needed
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = json.loads(raw)
        print(f"    Metrics extracted: {len(parsed)}")
        for m in parsed[:3]:
            print(f"      - {m.get('title')}: {m.get('current')} [{m.get('status')}]")
    except Exception as e:
        print(f"    EXCEPTION — {e}")

print("\n[Done]")
