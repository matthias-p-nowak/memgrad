
format:
    ufmt format src tests

test:
    pytest 

venv:
    python -m venv .venv

build:
    python -m build 

clean:
    rm -frv dist **/*egg-info .pytest_cache