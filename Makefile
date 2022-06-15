SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."
	@echo "test    : test code, data, and models."

# Styling
.PHONY: style
style:
	black .
	flake8 --exit-zero
	isort .

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

# Test
.PHONY: test
test:
	pytest tests
	cd tests && great_expectations checkpoint run math_problems
