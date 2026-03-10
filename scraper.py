#!/usr/bin/env python3
"""
YouTube AI Caption Scraper
Downloads YouTube audio via yt-dlp and transcribes it using faster-whisper.
"""

import sys
import os
import argparse
import tempfile

import yt_dlp
from faster_whisper import WhisperModel


def download_audio(url: str, output_dir: str) -> tuple[str, str]:
    """Download audio-only from a YouTube URL using yt-dlp.

    Returns (video_title, path_to_wav_file).
    """
    ydl_opts = {
        "format": "bestaudio/best",
        # Use video ID as filename to avoid special character issues
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        title = info.get("title", video_id)
        audio_path = os.path.join(output_dir, f"{video_id}.wav")
        return title, audio_path


def transcribe(audio_path: str, model: WhisperModel) -> str:
    """Transcribe an audio file using faster-whisper.

    Returns the full transcript as a plain string.
    """
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


def sanitize_filename(name: str) -> str:
    """Strip characters that are invalid in filenames."""
    keep = set(" -_.()")
    sanitized = "".join(c for c in name if c.isalnum() or c in keep).strip()
    return sanitized or "transcript"


def process_urls(urls: list[str], output_dir: str, model_size: str) -> None:
    print(f"\nLoading Whisper model '{model_size}' (downloads on first run)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("Model ready.\n")

    os.makedirs(output_dir, exist_ok=True)

    success, failed = 0, 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")

            try:
                print("  Downloading audio...")
                title, audio_path = download_audio(url, tmp_dir)
                print(f"  Title: {title}")

                print("  Transcribing...")
                transcript = transcribe(audio_path, model)

                # Write transcript file
                out_name = sanitize_filename(title) + ".txt"
                out_path = os.path.join(output_dir, out_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"URL:   {url}\n")
                    f.write("-" * 60 + "\n\n")
                    f.write(transcript + "\n")

                print(f"  Saved: {out_path}")
                success += 1

            except Exception as exc:
                print(f"  ERROR: {exc}")
                failed += 1

            finally:
                # Clean up the downloaded audio file
                if "audio_path" in dir() and os.path.exists(audio_path):
                    os.remove(audio_path)

    print(f"\nDone. {success} succeeded, {failed} failed.")
    print(f"Transcripts are in '{output_dir}/'")


def collect_urls(inputs: list[str]) -> list[str]:
    """Accept raw URLs or paths to .txt files containing one URL per line."""
    urls = []
    for item in inputs:
        if item.endswith(".txt") and os.path.isfile(item):
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        else:
            urls.append(item)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos and generate AI captions via faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraper.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python scraper.py URL1 URL2 URL3 -o my_transcripts
  python scraper.py urls.txt -m small
  python scraper.py urls.txt URL1 -o output -m medium

Model sizes (accuracy vs. speed trade-off):
  tiny   ~39M  params  fastest, least accurate
  base   ~74M  params  good balance (default)
  small  ~244M params  better accuracy
  medium ~769M params  high accuracy, slow
  large-v3 ~1.5B params  best accuracy, very slow
        """,
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="YouTube URL(s) and/or .txt file(s) with one URL per line",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        metavar="DIR",
        help="Directory to save transcript .txt files (default: output)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )

    args = parser.parse_args()
    urls = collect_urls(args.input)

    if not urls:
        print("No URLs found. Provide YouTube URLs or a .txt file with one URL per line.")
        sys.exit(1)

    print(f"Found {len(urls)} URL(s) to process.")
    process_urls(urls, args.output, args.model)


if __name__ == "__main__":
    main()
