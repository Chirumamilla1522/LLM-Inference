#!/usr/bin/env python3
"""Sync _source assets into per-article image folders."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from medium_image_layout import (  # noqa: E402
    ARTICLES,
    IMG,
    MANIFEST,
    SOURCE,
    migrate_legacy_into_source,
    resolve_article,
    write_article_readmes,
)


def find_source(src_rel: str) -> Path | None:
    candidates = [
        SOURCE / src_rel,
        IMG / src_rel,  # legacy flat layout
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def organize(article: str | None = None) -> None:
    migrate_legacy_into_source()
    slugs = [article] if article else ARTICLES
    for slug in slugs:
        dest_dir = IMG / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src_rel, dest_name in MANIFEST[slug]:
            src = find_source(src_rel)
            dest = dest_dir / dest_name
            if src is None:
                if dest.exists():
                    print(f"keep existing {slug}/{dest_name} (no source)")
                else:
                    print(f"MISSING {src_rel} -> {slug}/{dest_name}")
                continue
            shutil.copy2(src, dest)
            print(f"{slug}/{dest_name}")
    write_article_readmes(article)
    print(f"Done. Article folders under {IMG}/")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", "-a", default=None, help="Only sync one article (e.g. 00 or 00-introduction)")
    args = parser.parse_args()
    organize(resolve_article(args.article))


if __name__ == "__main__":
    main()
