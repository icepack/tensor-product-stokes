
results-2d.pdf: results-2d.json make_plots.py
	python make_plots.py --input $< --output $@

results-2d.json: lbb-stability.py
	for element in mini taylor-hood crouzeix-raviart; do \
		python lbb-stability.py --dimension 2 --element $$element --output $@ ; \
	done

results-3d.json: lbb-stability.py
	@for element in mini taylor-hood crouzeix-raviart; do		\
		for num_layers in 1 2 3; do 				\
			for vdegree in 1 2 3 4; do 			\
				python lbb-stability.py 		\
					--dimension 3 			\
					--element $$element 		\
					--num-cells-min 4 		\
					--num-cells-max 32 		\
					--num-cells-step 2 		\
					--num-layers $$num_layers	\
					--vdegree $$vdegree 		\
					--output $@ ; 			\
			done						\
		done							\
	done
