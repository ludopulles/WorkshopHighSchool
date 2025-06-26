.PHONY: tikz exercises_math.pdf
all: tikz exercises_math.pdf
tikz:
	make -C tikz
exercises_math.pdf:
	pdflatex exercises_math
