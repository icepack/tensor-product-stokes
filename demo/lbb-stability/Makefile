all: taylor-hood-2d.png taylor-hood-3d.png

taylor-hood-2d.png: lbb-stability.py
	python3 lbb-stability.py --dimension 2 --output $@

taylor-hood-3d.png: lbb-stability.py
	python3 lbb-stability.py --dimension 3 --output $@
