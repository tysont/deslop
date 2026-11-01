# ABOUTME: Builds tests/fixtures/registers: four register pairs (essay, explainer, product,
# ABOUTME: fiction), each a public-domain original plus a 3-5 document corpus for a small voice.
# ruff: noqa: E501  (prose literal and source tables are wider than the code limit)
from __future__ import annotations

import re
import sys
from pathlib import Path

from deslop.corpus.normalize import normalize

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "registers"
GUTENBERG = "https://www.gutenberg.org/ebooks/{id}"
LICENSE = "Public domain (Project Gutenberg)"

# (register, name, gutenberg id, author, title, start marker, end marker, word cap)
CORPORA = {
    "essay": [
        (
            10343,
            "Charles Lamb",
            "Essays of Elia",
            "THE SOUTH-SEA HOUSE",
            "THE OLD AND THE NEW SCHOOLMASTER",
            12000,
        ),
        (
            3020,
            "William Hazlitt",
            "Table-Talk",
            "ON THE PLEASURE OF PAINTING",
            "ON PEOPLE WITH ONE IDEA",
            12000,
        ),
        (8092, "G. K. Chesterton", "Tremendous Trifles", "The Secret of a Train", None, 12000),
        (16643, "Ralph Waldo Emerson", "Essays, First Series", "COMPENSATION.", None, 12000),
    ],
    "explainer": [
        (
            16729,
            "Thomas Henry Huxley",
            "Lay Sermons, Addresses and Reviews",
            "ON A PIECE OF CHALK.",
            "GEOLOGICAL CONTEMPORANEITY AND PERSISTENT TYPES OF LIFE.",
            12000,
        ),
        (
            24527,
            "John Tyndall",
            "Fragments of Science",
            "I. THE CONSTITUTION OF NATURE.",
            "II. RADIATION.",
            12000,
        ),
        (
            30155,
            "Albert Einstein (tr. Lawson)",
            "Relativity: The Special and General Theory",
            "PHYSICAL MEANING OF GEOMETRICAL PROPOSITIONS",
            None,
            12000,
        ),
        (
            28553,
            "Archibald Williams",
            "How It Works, Chapters V and VI",
            "WHAT IS ELECTRICITY?",
            "Chapter VII.",
            12000,
        ),
    ],
    "product": [
        (
            44556,
            "Blackie & Son",
            "Books for Young People, Catalogue 1898",
            "^BY G. A. HENTY.",
            None,
            9000,
        ),
        (
            45433,
            "S. W. Partridge & Co.",
            "Catalogue of Popular Illustrated Books",
            "^Illustrated",
            None,
            9000,
        ),
        (
            48619,
            "Blackie & Son",
            "Illustrated Story Books Catalogue, 1889",
            "^*G. A. HENTY*",
            None,
            9000,
        ),
        (54213, "Sears, Roebuck and Co.", "Military Equipment, 1917", "^Army Officers", None, 9000),
    ],
    "fiction": [
        (
            2776,
            "O. Henry",
            "The Four Million",
            "THE GIFT OF THE MAGI",
            "MAMMON AND THE ARCHER",
            10000,
        ),
        (
            1429,
            "Katherine Mansfield",
            "The Garden Party, and Other Stories",
            "And after all the weather was ideal",
            "The Daughters of the Late Colonel",
            10000,
        ),
        (
            13415,
            "Anton Chekhov (tr. Garnett)",
            "The Lady with the Dog and Other Stories",
            "THE LADY WITH THE DOG",
            "A DOCTOR'S VISIT",
            10000,
        ),
        (2814, "James Joyce", "Dubliners", "THE SISTERS", "AFTER THE RACE", 10000),
    ],
}

# (register, gutenberg id, author, title, start marker, end marker, word cap, note)
ORIGINALS = {
    "essay": (
        386,
        "Robert Louis Stevenson",
        "An Apology for Idlers (Virginibus Puerisque)",
        "AN APOLOGY FOR IDLERS",
        "EL DORADO",
        1000,
        "the opening of the essay",
    ),
    "explainer": (
        65307,
        "Edwin Sharpe Grew",
        "The Romance of Modern Geology, Chapter I: The Building of the Earth",
        "^Everybody who has ever been to the coast",
        None,
        1050,
        "the opening of the first chapter; chosen by `deslop lab fixture-check` over Williams "
        "and Munro (lowest recitation); the Faraday lecture it replaces recited at 4.9%",
    ),
    "fiction": (
        269,
        "Saki (H. H. Munro)",
        "The Open Window (Beasts and Super-Beasts)",
        "THE OPEN WINDOW",
        "THE COBWEB",
        960,
        "the opening of the story",
    ),
}

PRODUCT_ORIGINAL = """---
title: Field Kettle, 1.2 litre
author: deslop (written for this fixture)
license: written for deslop by its maintainer; use freely
note: Written by the deslop maintainer for this fixture set, not taken from any published source.
register: product
---

# Field Kettle, 1.2 litre

A kettle for a stove you carry. It holds 1.2 litres, enough for two mugs of tea and a pot of porridge, and it weighs 310 grams empty. The body is hard-anodised aluminium, 0.8 mm thick, with a rolled lip so it pours without dribbling down the side. The lid sits in a groove and stays put when you tip the kettle to pour.

The handle folds flat against the body and locks open with a steel pin. It is wrapped in silicone to the knuckle, so you can lift the kettle off a gas burner without a glove. The spout has a hinged cap that whistles at a rolling boil; take the cap off and it pours faster, for filling a bottle.

On a 2,000-watt canister stove, 1 litre of water at 15 degrees boils in four minutes and ten seconds at sea level. At 3,000 metres it took us five minutes and a half. The base is ribbed, which spreads the heat and keeps the kettle from rocking on a small burner head. It will not work on an induction hob.

The kettle nests inside the 1.8 litre pot from the same range, and a 230 gram gas canister fits inside the kettle, with the lid on. A mesh bag is included; it is not a dry bag and will not keep soot off your sleeping bag, so pack the kettle in the bag and the bag in an outside pocket.

Wash it by hand. The anodised finish takes scratches from a steel scourer, and a dishwasher will dull it. Do not boil the kettle dry: the handle wrap is rated to 230 degrees and the aluminium will warp above that.

Two years' guarantee against faults in materials and manufacture. Dents from drops and rocks are not faults. If the lid or the handle pin goes missing, both are sold separately at two pounds each, post paid.

Sizes: 1.2 litre (this page), 1.8 litre, 2.5 litre. Colours: grey, green. Made in Portugal.
"""


def slice_between(text: str, start: str, end: str | None, cap: int) -> str:
    """The text from `start` (its last occurrence, so a table of contents is skipped; a
    leading ^ asks for the first) to `end`, as unwrapped paragraphs, cut at a paragraph
    boundary under `cap` words. A short all-capitals paragraph becomes a heading (the first
    a title, the rest sections)."""
    i = text.find(start[1:]) if start.startswith("^") else text.rfind(start)
    start = start.lstrip("^")
    if i < 0:
        raise SystemExit(f"start marker not found: {start!r}")
    j = text.find(end, i + len(start)) if end else -1
    body = text[i : j if j > 0 else len(text)]
    paragraphs = []
    words = 0
    for chunk in re.split(r"\n\s*\n", body):
        para = " ".join(line.strip() for line in chunk.splitlines() if line.strip())
        if not para:
            continue
        n = len(para.split())
        if paragraphs and words + n > cap:
            break
        paragraphs.append(para)
        words += n
    for k, para in enumerate(paragraphs):
        if para.isupper() and len(para.split()) <= 8:
            paragraphs[k] = ("# " if k == 0 else "## ") + para.rstrip(".").title()
    return "\n\n".join(paragraphs) + "\n"


def frontmatter(**fields: str) -> str:
    lines = ["---"] + [f"{k}: {v}" for k, v in fields.items()] + ["---", ""]
    return "\n".join(lines)


def load(src: Path, gid: int) -> str:
    raw = (src / f"pg{gid}.txt").read_text(encoding="utf-8", errors="ignore")
    text = normalize(raw, gutenberg=True)
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"\[\d{1,3}\]", "", text)  # footnote markers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main(src: Path, only: str | None = None) -> None:
    for register, docs in CORPORA.items():
        if only and register != only:
            continue
        corpus = OUT / register / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        for gid, author, title, start, end, cap in docs:
            text = load(src, gid)
            body = slice_between(text, start, end, cap)
            slug = re.sub(r"[^a-z0-9]+", "_", author.split("(")[0].strip().lower()).strip("_")
            path = corpus / f"{slug}_{gid}.txt"
            path.write_text(
                frontmatter(
                    author=author, title=title, source=GUTENBERG.format(id=gid), license=LICENSE
                )
                + body,
                encoding="utf-8",
            )
            print(f"{register}: {path.name} {len(body.split())} words")
        if register in ORIGINALS:
            gid, author, title, start, end, cap, note = ORIGINALS[register]
            body = slice_between(load(src, gid), start, end, cap)
            (OUT / register / "original.md").write_text(
                frontmatter(
                    title=title,
                    author=author,
                    source=GUTENBERG.format(id=gid),
                    license=LICENSE,
                    note=f"Excerpt: {note}, cut at a paragraph boundary near {cap} words.",
                    register=register,
                )
                + body,
                encoding="utf-8",
            )
        else:
            (OUT / register / "original.md").write_text(PRODUCT_ORIGINAL, encoding="utf-8")
        print(
            f"{register}: original {len((OUT / register / 'original.md').read_text().split())} words"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), *sys.argv[2:3])
