
format:
    ufmt format memgrad tests

test:
    .venv/bin/pytest 

venv:
    python -m venv .venv

build:
    python -m build 

prepare:
    pip install -e .

clean:
    rm -frv dist **/*egg-info .pytest_cache **/__pycache__ *.egg-info