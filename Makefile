
tensor-product-stokes.pdf: tensor-product-stokes.tex tensor-product-stokes.bib
	pdflatex tensor-product-stokes.tex
	bibtex tensor-product-stokes
	pdflatex tensor-product-stokes.tex
	pdflatex tensor-product-stokes.tex
