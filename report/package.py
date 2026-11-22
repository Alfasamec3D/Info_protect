"""Create the public replication archive with checksums for its files."""

import argparse
import hashlib
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL = ("README.md", "LICENSE", "pyproject.toml", "constraints-study.txt")
SECTIONS = {
    "configs": {".json"},
    "src": {".py"},
    "tests": {".py"},
    "report": {".py", ".md", ".csv", ".docx", ".pdf", ".pptx"},
    "results": {".csv", ".json", ".png"},
}


def contents():
    yield from (ROOT / name for name in TOP_LEVEL)
    for directory, suffixes in SECTIONS.items():
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix in suffixes:
                yield path


def public_bytes(path, name):
    data = path.read_bytes()
    if name == "report/requirements.md":
        text = data.decode("utf-8")
        return re.sub(r"\[рабочая программа дисциплины\]\(<\.\./[^>]+\.docx>\)",
                      "рабочая программа дисциплины «Защита информации» (МФТИ)",
                      text).encode("utf-8")
    return data


def write_file(zipfile, name, data):
    item = ZipInfo(name)
    item.compress_type = ZIP_DEFLATED
    zipfile.writestr(item, data)


def create(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with ZipFile(destination, "w") as archive:
        for path in sorted(contents()):
            name = path.relative_to(ROOT).as_posix()
            data = public_bytes(path, name)
            write_file(archive, name, data)
            manifest.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
        write_file(archive, "SHA256SUMS", "".join(manifest).encode("utf-8"))
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"{args.output.resolve()} ({create(args.output)} files plus SHA256SUMS)")


if __name__ == "__main__":
    main()
