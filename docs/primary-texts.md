# Primary texts

A primary text is a source a rewrite may quote or cite: scripture, a dialogue, a statute, a
novel. `deslop apply` uses it for one thing, verification: an added quotation must appear in
it verbatim, and an added citation must resolve to a labelled span of it. deslop does not
know what any text is or what its citations mean. It knows three files.

## The format

1. **The text**: a plain-text or markdown file. Nothing in it is special; a header comment
   is fine.
2. **A sidecar** `<name>.locs.json` beside it (optional): a list of labelled spans, as
   character offsets into the text file.

   ```json
   {"locations": [
     {"ref": "Book 6:5", "start": 191, "end": 247},
     {"ref": "6:5", "start": 191, "end": 247}
   ]}
   ```

   A span may carry several labels (one entry each). Labels are matched case-insensitively
   with periods and trailing punctuation dropped, so `Bk. 6:5` finds `Bk 6:5`. Without a
   sidecar the text still verifies quotations, and nothing resolves.
3. **A citation scheme** per text, in `voice.yaml` or on the command line: a regex for what
   a citation of this text looks like, so deslop can recognise the string without knowing
   what it means.

   ```yaml
   primary_texts:
     - path: primary/book.md          # relative to the voice file
       title: Book
       schemes:
         - name: verse
           pattern: '\b(?:Book|Bk)\.?\s+\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?'
           bare_pattern: '(?<![\d:])\d{1,3}:\d{1,3}(?:[-–]\d{1,3})?(?![\d:])'
   ```

   `pattern` matches a citation that names the text; `bare_pattern` (optional) matches a
   bare location whose text is known from context. On the command line the same object is
   `--citation-scheme '{"name": "verse", "pattern": "...", "bare_pattern": "..."}'`, applied
   to every `--primary-text` given with it.

## Resolution

Resolution is three steps and nothing else: a scheme's regex matches the citation string;
the string, canonicalised, is looked up among the sidecar's labels; the label's span is the
answer. A range (`6:5-6`, `372c-e`) resolves by the part before the dash. A citation no
scheme recognises is listed as unverifiable and never rejected. A citation a scheme
recognises but no label answers is a rejection: declaring a scheme for a text promises that
its citations can be checked, so a sidecar that lacks those labels makes every such citation
fail. That is deliberate. If an edition carries no page numbers, say so by declaring the
scheme without labels, and any page citation a rewrite invents is caught.

Quotations are compared after normalisation (lower case, punctuation dropped, single
spaces). A quotation not found verbatim is rejected, and the report shows the nearest run
of the text with its location, so another translation's wording of the same passage is
visible as such.

## Two examples

**Scripture.** One file per book, a sidecar entry per verse with the labels the writer might
use (`Genesis 6:5`, `Gen 6:5`, `6:5`), and a verse scheme with a `bare_pattern` so `6:5`
resolves when the book is known from context.

**A dialogue cited by page and section.** The text file holds the dialogue; the sidecar
labels whatever the edition can locate. If the edition carries the traditional page
numbers, label them (`369b`) and declare a scheme such as
`'\b\d{3}[a-e](?:[-–](?:\d{3})?[a-e])?\b'`. If it does not, label what it has (chapters,
paragraphs) and still declare the page scheme: page citations are then recognised and
rejected, which is what you want from a text that cannot confirm them.

## What apply does with the result

A resolved citation and a verbatim quotation are permitted and not material: they are
listed under "verified additions" with their locations. A sentence that merely names a
declared text, and a claim covered by a verified quotation, are cleared with the reason.
A fact the input lacks remains a material addition unless `--allow-primary-facts` is on,
which permits a claim whose sentence cites a resolved location and whose numbers and names
all occur in that span. Converters from an edition's own layout into this format belong
with the edition, not in deslop.
