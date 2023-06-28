
cylinder_flow.pdf: crouzeix-raviart.h5
	python make_plots.py --input $< --output $@

mini.h5: cylinder_flow.py
	python cylinder_flow.py --element mini --output $@

taylor-hood.h5: cylinder_flow.py
	python cylinder_flow.py --element taylor-hood --output $@

crouzeix-raviart.h5: cylinder_flow.py
	python cylinder_flow.py --element crouzeix-raviart --output $@
