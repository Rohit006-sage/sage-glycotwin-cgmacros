"""Check local documentation links and dashboard wording without private data."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "EVALUATION.md",
    ROOT / "docs" / "MODEL_CARD.md",
    ROOT / "docs" / "JURY_QA.md",
    ROOT / "docs" / "DEMO_SCRIPT.md",
]
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def heading_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
        for heading in HEADING_PATTERN.findall(text)
    }


def check_links() -> list[str]:
    issues: list[str] = []
    for source in MARKDOWN_FILES:
        text = source.read_text(encoding="utf-8")
        for link in LINK_PATTERN.findall(text):
            if link.startswith(("http://", "https://", "#")):
                continue
            target, _, fragment = link.partition("#")
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                issues.append(f"{source.relative_to(ROOT)} -> {link}: missing file")
                continue
            if fragment and resolved.suffix.lower() in {".md", ".mdx"}:
                if fragment not in heading_ids(resolved):
                    issues.append(f"{source.relative_to(ROOT)} -> {link}: missing anchor")
    return issues


def check_dashboard_wording() -> list[str]:
    issues: list[str] = []
    dashboard_files = [
        ROOT / "serve_cgmacros_dashboard.py",
        ROOT / "cgmacros_dashboard.html",
        ROOT / "docs" / "DEMO_SCRIPT.md",
    ]
    stale_phrases = (
        "pending a purged-boundary rerun",
        "creates an early-label cutoff concern",
        "because early `t+120` training labels can cross the 72-hour cutoff",
    )
    for path in dashboard_files:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in stale_phrases:
            if phrase in text:
                issues.append(f"{path.relative_to(ROOT)}: stale dashboard wording: {phrase}")
    return issues


def main() -> None:
    issues = check_links() + check_dashboard_wording()
    if issues:
        raise SystemExit("Documentation checks failed:\n- " + "\n- ".join(issues))
    print("Documentation links, anchors, and dashboard wording are valid.")


if __name__ == "__main__":
    main()
