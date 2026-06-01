# Build the survey PDFs from LaTeX.
# Requires a TeX Live (or similar) toolchain with latexmk + biber.
DOCS = MLConcepts 01-landscape \
       02-group1-circuit-algebra 03-group2-superposition-geometry \
       04-group3-dictionary-learning 05-group4-circuit-tracing \
       06-group5-representation-geometry 07-group6-epistemics \
       08-open-problems
PDFS = $(DOCS:=.pdf)

all: $(PDFS)

%.pdf: %.tex references.bib
	latexmk -pdf -interaction=nonstopmode $<

clean:        ## remove build artifacts, keep PDFs
	latexmk -c

distclean:    ## remove build artifacts and PDFs
	latexmk -C

.PHONY: all clean distclean
