.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib neurips_2025.sty
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc \
	      main.fdb_latexmk main.fls main.synctex.gz
