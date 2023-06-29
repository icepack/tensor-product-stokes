
tensor-product-stokes.pdf: tensor-product-stokes.tex tensor-product-stokes.bib
	cd demo/flow-around-cylinder && make cylinder_flow.pdf && cd ../../
	cd demo/lbb-stability && make results-2d.pdf && cd ../../
	pdflatex tensor-product-stokes.tex
	bibtex tensor-product-stokes
	pdflatex tensor-product-stokes.tex
	pdflatex tensor-product-stokes.tex
