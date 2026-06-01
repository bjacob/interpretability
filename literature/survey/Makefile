# Build the survey PDFs from LaTeX.
# Requires a TeX Live (or similar) toolchain with latexmk + biber.
DOCS = MLConcepts 01-landscape
PDFS = $(DOCS:=.pdf)

all: $(PDFS)

%.pdf: %.tex references.bib
	latexmk -pdf -interaction=nonstopmode $<

clean:        ## remove build artifacts, keep PDFs
	latexmk -c

distclean:    ## remove build artifacts and PDFs
	latexmk -C

.PHONY: all clean distclean
