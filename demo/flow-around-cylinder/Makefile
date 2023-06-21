all: mini.h5 taylor-hood.h5 crouzeix-raviart.h5

mini.h5: cylinder_flow.py
	python cylinder_flow.py --element mini --output $@

taylor-hood.h5: cylinder_flow.py
	python cylinder_flow.py --element taylor-hood --output $@

crouzeix-raviart.h5: cylinder_flow.py
	python cylinder_flow.py --element crouzeix-raviart --output $@
