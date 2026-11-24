"""Render project PDFs to PNGs using the already installed PyMuPDF package."""

import sys
from pathlib import Path

import fitz


def render(pdf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    print(f"{pdf_path.name}: {len(document)} page(s)")
    for index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        output = output_dir / f"{pdf_path.stem}-page-{index}.png"
        pixmap.save(output)
        print(output)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target_dir = root / "artifacts" / "pdf_render"
    for pdf in sorted((root / "data" / "demo_pdfs").glob("*.pdf")):
        render(pdf, target_dir)
