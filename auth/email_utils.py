"""
email_utils.py — Calls the Vercel email microservice to send OTP emails.
The backend never talks to Resend directly; it delegates to the TS service.
"""
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL", "http://localhost:3000")


def send_otp_email(email: str, otp: str, name: str | None = None) -> None:
    """
    POST the OTP to the email microservice.
    Raises RuntimeError if the microservice returns an error or is unreachable.
    """
    url = f"{EMAIL_SERVICE_URL.rstrip('/')}/api/send-otp"
    payload: dict = {"email": email, "otp": otp}
    if name:
        payload["name"] = name

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)

        if response.status_code != 200:
            error_detail = response.json().get("error", "Unknown error from email service")
            raise RuntimeError(f"Email service error: {error_detail}")

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Could not reach the email service at {url}. "
            f"Is it running? Detail: {exc}"
        ) from exc
