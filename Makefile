
default:
	@echo "\"make upload\"?"

README.rst: README.md
	pandoc README.md -o README.rst
	python setup.py check -r -s || exit 1

upload: setup.py README.rst
	python setup.py sdist upload --sign

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/ MANIFEST

black:
	black setup.py pynosh/ test/*.py

lint:
	black --check setup.py pynosh/ test/*.py
	flake8 setup.py pynosh/ test/*.py
