
format:
    ufmt format memgrad tests

test:
    .venv/bin/pytest 

venv:
    python -m venv .venv

build:
    python -m build 

clean:
    rm -frv dist **/*egg-info .pytest_cache