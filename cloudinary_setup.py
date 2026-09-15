# -*- coding: utf-8 -*-
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cloudinary
import cloudinary.uploader
import cloudinary.api

# ─────────────────────────────────────────────
# STEP 1: Configure Cloudinary (from environment variables)
# Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET in your .env
# ─────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

print("=" * 55)
print("  CognoCareAI — Cloudinary Integration Setup")
print("=" * 55)

# ─────────────────────────────────────────────
# STEP 2: Upload a sample image from Cloudinary demo
# Supported in your app: PDF, JPG, PNG
# ─────────────────────────────────────────────
SAMPLE_IMAGE_URL = "https://res.cloudinary.com/demo/image/upload/sample.jpg"

print("\n[1] Uploading sample image...")

upload_result = cloudinary.uploader.upload(
    SAMPLE_IMAGE_URL,
    folder="cognocareai/uploads",   # organises uploads in your Cloudinary account
    resource_type="image"           # change to "raw" for PDF uploads
)

secure_url = upload_result.get("secure_url")
public_id  = upload_result.get("public_id")

print(f"    [OK] Upload successful!")
print(f"    Secure URL : {secure_url}")
print(f"    Public ID  : {public_id}")

# ─────────────────────────────────────────────
# STEP 3: Fetch image metadata
# ─────────────────────────────────────────────
print("\n[2] Fetching image metadata...")

details = cloudinary.api.resource(public_id)

width   = details.get("width")
height  = details.get("height")
fmt     = details.get("format")
size    = details.get("bytes")

print(f"    Width      : {width}px")
print(f"    Height     : {height}px")
print(f"    Format     : {fmt}")
print(f"    File size  : {size} bytes ({round(size / 1024, 1)} KB)")

# ─────────────────────────────────────────────
# STEP 4: Generate a transformed image URL
#   f_auto → Cloudinary automatically picks the best format
#            for the user's browser (e.g. WebP, AVIF)
#   q_auto → Cloudinary automatically selects the best
#            quality level to reduce file size while keeping
#            the image visually identical
# ─────────────────────────────────────────────
print("\n[3] Generating optimised transformed URL...")

transformed_url = cloudinary.CloudinaryImage(public_id).build_url(
    transformation=[
        {"fetch_format": "auto"},   # f_auto — best format for the browser
        {"quality": "auto"}         # q_auto — optimal quality/size balance
    ]
)

print("\n" + "=" * 55)
print("  Done! Click the link below to see the optimised version.")
print("  Check the size and format in your browser DevTools (F12 > Network).")
print("=" * 55)
print(f"\n  --> Transformed URL:\n  {transformed_url}\n")
