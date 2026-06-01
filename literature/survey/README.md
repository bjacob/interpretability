# literature/survey/

Surveys synthesised from reading the papers in `../papers/`, written in **LaTeX** so the
heavy use of mathematics renders properly (GitHub's Markdown math support was too limited).
Citations point at the **original source URLs** via `references.bib`, not at our local mirror.

## Documents
- **`MLConcepts.tex`** → **`MLConcepts.pdf`** — a standing primer on the ML vocabulary for a
  mathematician / systems engineer who is not an ML specialist (transformer architecture in
  spaces-and-maps terms, an adjoint-vs-dual notational convention, the interpretability lexicon).
- **`01-landscape.tex`** → **`01-landscape.pdf`** — the landscape survey: six methodological
  groups, the shared mathematical substrate, where the math converges vs. fundamentally diverges,
  and an agenda of candidate mathematical blind spots.
- **`references.bib`** — bibliography; one entry per source, keyed by the paper "stem", each
  pointing at its original URL.

The compiled **PDFs are committed on purpose** so the surveys are readable directly on GitHub.

## Building
Requires a TeX Live (or similar) toolchain with `latexmk` and `biber`:
```sh
make            # build both PDFs
make clean      # remove LaTeX byproducts, keep PDFs
make distclean  # remove byproducts and PDFs
```
LaTeX byproducts (`*.aux`, `*.bbl`, …) are git-ignored; only the `.tex`, `.bib`, `Makefile`,
this README, and the `.pdf` outputs are tracked.

## Conventions
- Numbered surveys (`NN-topic.tex`) are sequential; later ones may revise earlier ones.
- Adjoint `$W^{*}$` vs. dual map `$W^{\vee}$` are kept distinct (see `MLConcepts.tex`, the
  notational-convention box); a bare transpose `$^\top$` appears only as a literal matrix
  transpose in expressions explicitly marked "coordinates"/"matrix picture".
