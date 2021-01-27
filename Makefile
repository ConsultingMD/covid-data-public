.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install --upgrade -r requirements.txt -r requirements_test.txt
	pre-commit install
	nbstripout --install

unittest:
	pytest -n 2 tests/

lint:
	pytest --pylint -m pylint --pylint-error-types=EF .

test: fmt lint unittest

fmt:
	black .
