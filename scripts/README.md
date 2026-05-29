# scripts/

Reusable scripts for this repo.

## render_survey_html.py
Renders `literature/survey/*.md` to printable, math-typeset HTML (LaTeX via MathJax).

```sh
pip install markdown            # one-time dependency (a venv is fine)
python3 scripts/render_survey_html.py [OUTPUT_DIR]   # default: <tmp>/survey-html
```

Open the resulting `.html` in a browser and print from there. Math is typeset by MathJax
loaded from a CDN, so first view needs network; page styling is inlined.

**Generated HTML is deliberately not committed** — it would drift from the `.md` source.
The markdown files are the single source of truth; regenerate the HTML on demand.
