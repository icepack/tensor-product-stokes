
results-2d.pdf: results-2d.json make_plots.py
	python make_plots.py --input $< --output $@

results-2d.json: lbb-stability.py
	python lbb-stability.py --dimension 2 --element mini --output $@
	python lbb-stability.py --dimension 2 --element taylor-hood --output $@
	python lbb-stability.py --dimension 2 --element crouzeix-raviart --output $@
