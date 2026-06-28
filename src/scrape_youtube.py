import argparse
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit query 
DEFAULT_QUERIES = [
    "bad squat form",
]
DEFAULT_MAX_RESULTS = 10
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "youtube_query" / "videos"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "youtube_query" / "downloaded.txt"


def normalize_source(text):
    text = text.strip()
    if not text or text.startswith("#"):
        return ""
    parts = text.split()
    if len(parts) == 2 and parts[0].lower() == "youtube":
        text = parts[1]
    if text.startswith(("http://", "https://")):
        return text
    return f"https://www.youtube.com/watch?v={text}"


def read_sources(path):
    return [url for url in (normalize_source(line) for line in Path(path).read_text().splitlines()) if url]


def query_source(query, max_results):
    return f"ytsearch{max_results}:{query}"


def yt_dlp_command(urls, out_dir, archive, write_info_json=True):
    out_dir = Path(out_dir)
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--no-playlist",
        "--download-archive",
        str(archive),
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "-o",
        str(out_dir / "%(id)s.%(ext)s"),
    ]
    if write_info_json:
        cmd.append("--write-info-json")
    return [*cmd, *urls]


def download_youtube(urls, out_dir, archive, dry_run=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(archive).parent.mkdir(parents=True, exist_ok=True)
    cmd = yt_dlp_command(urls, out_dir, archive)
    if dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd, check=False).returncode


def self_test():
    assert normalize_source("youtube abc123") == "https://www.youtube.com/watch?v=abc123"
    assert normalize_source("abc123") == "https://www.youtube.com/watch?v=abc123"
    assert normalize_source("https://youtu.be/abc123") == "https://youtu.be/abc123"
    assert query_source("bad squat form", 5) == "ytsearch5:bad squat form"
    cmd = yt_dlp_command(["https://youtu.be/abc123"], "videos", "videos/downloaded.txt")
    assert "yt-dlp" == cmd[0] and "--download-archive" in cmd
    assert DEFAULT_QUERIES
    print("self-test ok")


def parse_args():
    parser = argparse.ArgumentParser(description="Download listed YouTube videos for later pose-feature extraction.")
    parser.add_argument("urls", nargs="*", help="YouTube URLs or video IDs.")
    parser.add_argument("--query", action="append", default=[], help="YouTube search query. Can be repeated.")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--url-file", help="Text file with one URL/video ID per line.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        urls = [normalize_source(url) for url in args.urls]
        if args.url_file:
            urls.extend(read_sources(args.url_file))
        queries = args.query or DEFAULT_QUERIES
        urls.extend(query_source(query, args.max_results) for query in queries)
        urls = [url for url in urls if url]
        if not urls:
            raise SystemExit("No YouTube URLs, IDs, or --query provided.")
        raise SystemExit(download_youtube(urls, args.out_dir, args.archive, args.dry_run))
