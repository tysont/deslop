# ABOUTME: Corpus directory model: corpus.yaml manifest plus normalized text files.
# ABOUTME: Supports init, add (with dedupe), load, and share statistics.
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from deslop.corpus.dedupe import fingerprint, is_duplicate
from deslop.corpus.ingest import ingest
from deslop.nlp import count_words

MANIFEST = "corpus.yaml"


class DocumentEntry(BaseModel):
    path: str
    author: str
    title: str
    year: int | None = None
    weight: float = 1.0
    license: str = ""
    simhash: int | None = None
    words: int | None = None


class CorpusManifest(BaseModel):
    name: str
    language: str = "en"
    documents: list[DocumentEntry] = Field(default_factory=list)


class DocumentText(BaseModel):
    entry: DocumentEntry
    text: str

    @property
    def words(self) -> int:
        return count_words(self.text)


class AuthorShare(BaseModel):
    author: str
    words: int
    share: float
    documents: int


class CorpusStats(BaseModel):
    total_words: int
    authors: list[AuthorShare]
    over_cap: list[str]
    max_share: float


class DuplicateError(Exception):
    pass


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "document"


class Corpus:
    def __init__(self, root: Path, manifest: CorpusManifest):
        self.root = Path(root)
        self.manifest = manifest

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def init(cls, root: Path, name: str | None = None) -> Corpus:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if (root / MANIFEST).exists():
            return cls.load(root)
        corpus = cls(root, CorpusManifest(name=name or root.name))
        corpus.save()
        return corpus

    @classmethod
    def load(cls, root: Path) -> Corpus:
        root = Path(root)
        with (root / MANIFEST).open(encoding="utf-8") as fh:
            manifest = CorpusManifest.model_validate(yaml.safe_load(fh) or {})
        return cls(root, manifest)

    def save(self) -> None:
        data = self.manifest.model_dump(exclude_none=True)
        with (self.root / MANIFEST).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------ content
    @property
    def name(self) -> str:
        return self.manifest.name

    def documents(self) -> list[DocumentText]:
        docs = []
        for entry in self.manifest.documents:
            text = (self.root / entry.path).read_text(encoding="utf-8")
            docs.append(DocumentText(entry=entry, text=text))
        return docs

    def add(
        self,
        source: str,
        *,
        author: str,
        title: str | None = None,
        year: int | None = None,
        weight: float = 1.0,
        license: str = "",
    ) -> DocumentEntry:
        text, suggested = ingest(source)
        title = title or suggested
        fp = fingerprint(text)
        existing = [d.simhash for d in self.manifest.documents if d.simhash is not None]
        if is_duplicate(fp, existing):
            raise DuplicateError(f"{title!r} is a near-duplicate of an existing document")
        filename = f"{slugify(author.split()[-1])}_{slugify(title)}.txt"
        target = self.root / filename
        counter = 2
        while target.exists():
            target = self.root / f"{filename[:-4]}_{counter}.txt"
            counter += 1
        target.write_text(text, encoding="utf-8")
        entry = DocumentEntry(
            path=target.name,
            author=author,
            title=title,
            year=year,
            weight=weight,
            license=license,
            simhash=fp,
            words=count_words(text),
        )
        self.manifest.documents.append(entry)
        self.save()
        return entry

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for entry in sorted(self.manifest.documents, key=lambda d: d.path):
            h.update(entry.path.encode())
            h.update((self.root / entry.path).read_bytes())
        return h.hexdigest()[:16]

    def stats(self, max_share: float = 0.25) -> CorpusStats:
        by_author: dict[str, list[int]] = {}
        for doc in self.documents():
            by_author.setdefault(doc.entry.author, []).append(doc.words)
        total = sum(sum(v) for v in by_author.values())
        authors = [
            AuthorShare(
                author=a,
                words=sum(w),
                share=(sum(w) / total if total else 0.0),
                documents=len(w),
            )
            for a, w in sorted(by_author.items(), key=lambda kv: -sum(kv[1]))
        ]
        over = [a.author for a in authors if a.share > max_share]
        return CorpusStats(total_words=total, authors=authors, over_cap=over, max_share=max_share)
