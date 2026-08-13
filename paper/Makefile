.PHONY: all clean

all: main.pdf

main.pdf: main.tex sec_*.tex references.bib neurips_2025.sty
	tectonic -X compile main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc \
	      main.fdb_latexmk main.fls main.synctex.gz
