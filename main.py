#!/usr/bin/env python3
"""
FastAPI wrapper for the YouTube transcriber scraper.
"""

import tempfile
import asyncio
import base64
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

from scraper import download_audio, transcribe, get_channel_stats

app = FastAPI(title="YouTube Transcriber")
executor = ThreadPoolExecutor(max_workers=2)
model: WhisperModel = None


@app.on_event("startup")
async def startup():
    global model
    model = WhisperModel("base", device="cpu", compute_type="int8")


class TranscribeRequest(BaseModel):
    url: str
    cookies: str | None = None  # Base64-encoded Netscape-format cookie file contents


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_video(req: TranscribeRequest):
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, _process, req.url, req.cookies)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


def _process(url: str, cookies_b64: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        cookies_file = None
        if cookies_b64:
            cookies_file = os.path.join(tmp_dir, "cookies.txt")
            with open(cookies_file, "w") as f:
                f.write(base64.b64decode(cookies_b64).decode("utf-8"))

        title, audio_path, info = download_audio(url, tmp_dir, cookies_file)

        channel_url = info.get("channel_url", "")
        channel_stats = get_channel_stats(channel_url) if channel_url else {}

        transcript = transcribe(audio_path, model)

        stats = [
            {"video_description": info.get("description", "")},
            {"channel_description": channel_stats.get("channel_description", "")},
            {"subscribers": channel_stats.get("subscribers")},
            {"average_views": channel_stats.get("average_views")},
        ]

        return {"title": title, "url": url, "stats": stats, "transcript": transcript}
