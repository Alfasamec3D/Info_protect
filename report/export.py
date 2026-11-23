"""Export Office files to PDF and correct the report TOC from actual pagination."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from docx import Document

from build import SECTIONS


def page_map(pdf):
    pages = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).split("\f")
    mapping = {}
    for heading in SECTIONS:
        key = "".join(heading.upper().split())
        matches = [i + 1 for i, text in enumerate(pages) if i >= 3
                   and "".join(text[:500].upper().split()).startswith(key)]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify a unique page for {heading!r}: {matches}")
        mapping[heading] = matches[0]
    return mapping


def export(directory):
    # Separate LibreOffice profile avoids interference with the user's open Office files.
    with tempfile.TemporaryDirectory(prefix="puf-office-") as temp:
        def convert(names):
            subprocess.run(["libreoffice", f"-env:UserInstallation={Path(temp).as_uri()}",
                            "--headless", "--convert-to", "pdf", "--outdir", str(directory),
                            *[str(directory / name) for name in names]], check=True)
        convert(("report.docx", "slides.pptx"))
        for _ in range(3):
            mapping = page_map(directory / "report.pdf")
            doc = Document(directory / "report.docx")
            changed = False
            for paragraph in doc.paragraphs:
                title = paragraph.text.split("\t")[0]
                if title in mapping and "\t" in paragraph.text:
                    corrected = f"{title}\t{mapping[title]}"
                    if paragraph.text != corrected:
                        paragraph.text = corrected
                        changed = True
            if not changed:
                return mapping
            doc.save(directory / "report.docx")
            convert(("report.docx",))
        raise RuntimeError("Table of contents pagination did not stabilize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("report"))
    args = parser.parse_args()
    for heading, page in export(args.report.resolve()).items():
        print(f"{page}: {heading}")
