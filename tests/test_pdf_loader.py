"""Tests for PDF loader helper behavior."""

# OPTIMIZED_BY_CODEX_STEP_4
from pathlib import Path

from app.ingestion.loaders.pdf_loader import PDFLoader


def test_pdf_loader_scans_recursive_pdf_files(tmp_path: Path):
    root = tmp_path / "中文目录"
    nested = root / "sub"
    nested.mkdir(parents=True)

    (root / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "b.PDF").write_bytes(b"%PDF-1.4\n")
    (nested / "ignore.txt").write_text("not pdf", encoding="utf-8")

    loader = PDFLoader(pdf_dir=root, max_files=100, chunk_size=20, chunk_overlap=5)
    files = loader._scan_pdf_files()

    assert len(files) == 2
    assert all(f.suffix.lower() == ".pdf" for f in files)


def test_pdf_loader_split_text_respects_overlap():
    loader = PDFLoader(pdf_dir=Path("."), max_files=0, chunk_size=10, chunk_overlap=2)
    chunks = loader._split_text("abcdefghijklmnopqrstuvwxyz")

    assert len(chunks) >= 3
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ijkl")


# STEP_4_SUMMARY: Added PDF loader tests for recursive scan and chunk overlap splitting.
