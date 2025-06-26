texfiles := $(wildcard *.tex)
pdffiles := $(patsubst %.tex, %.pdf, $(texfiles))

all: $(pdffiles)
clean:
	rm *.log *.aux *.pdf *.gnuplot *.table

%.pdf: %.tex
	pdflatex -shell-escape $(patsubst %.pdf,%.tex,$@)
