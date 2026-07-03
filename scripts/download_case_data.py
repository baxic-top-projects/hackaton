from __future__ import annotations

import re
from pathlib import Path

import requests


PUBLIC_URL = "https://disk.yandex.ru/d/qE55fooRQGNVVA"
API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources"
DOWNLOAD_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
TARGET_DIR = Path("data/case_yandex")


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    root = requests.get(API_URL, params={"public_key": PUBLIC_URL, "limit": 100}, timeout=30)
    root.raise_for_status()
    start_path = root.json()["_embedded"]["items"][0]["path"]
    download_tree(start_path, TARGET_DIR)


def download_tree(remote_path: str, local_dir: Path) -> None:
    response = requests.get(
        API_URL,
        params={"public_key": PUBLIC_URL, "path": remote_path, "limit": 500},
        timeout=30,
    )
    response.raise_for_status()
    for item in response.json().get("_embedded", {}).get("items", []):
        target = local_dir / safe_name(item["name"])
        if item["type"] == "dir":
            target.mkdir(parents=True, exist_ok=True)
            download_tree(item["path"], target)
            continue
        if item["type"] != "file":
            continue
        href_response = requests.get(
            DOWNLOAD_URL,
            params={"public_key": PUBLIC_URL, "path": item["path"]},
            timeout=30,
        )
        href_response.raise_for_status()
        href = href_response.json().get("href")
        if not href:
            continue
        file_response = requests.get(href, timeout=120)
        file_response.raise_for_status()
        target.write_bytes(file_response.content)
        message = f"{target} ({len(file_response.content)} bytes)"
        print(message.encode("unicode_escape").decode("ascii"))


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\\\|?*]+', "_", name)


if __name__ == "__main__":
    main()
