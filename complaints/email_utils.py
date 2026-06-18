import os
import httpx
from dotenv import load_dotenv

load_dotenv()

EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL", "http://localhost:3000")

def send_complaint_email(email: str, category: str, description: str, name: str | None = None) -> None:
    """
    POST the complaint details to the email microservice.
    """
    url = f"{EMAIL_SERVICE_URL.rstrip('/')}/api/send-complaint"
    payload: dict = {"email": email, "category": category, "description": description}
    if name:
        payload["name"] = name

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)

        if response.status_code != 200:
            try:
                error_detail = response.json().get("error", "Unknown error from email service")
            except Exception:
                error_detail = f"HTTP {response.status_code}: {response.text}"
            print(f"Email service error: {error_detail}")

    except httpx.RequestError as exc:
        print(f"Could not reach the email service at {url}. Detail: {exc}")
