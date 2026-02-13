PROJECT_SOURCE_DIR ?= $(abspath ./)
PROJECT_NAME ?= $(shell basename $(PROJECT_SOURCE_DIR))
NUM_JOB ?= 8

all:
	@echo nothing special
.PHONY: all

clean:
	rm -rf build dist
.PHONY: clean

PYTHON ?= python3
build:
	$(PYTHON) -m pip install scikit_build_core pyproject_metadata pathspec pybind11
	CMAKE_BUILD_PARALLEL_LEVEL=$(NUM_JOB) $(PYTHON) -m pip install --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build -ve.
python_install:
	$(PYTHON) -m pip install . --verbose
python_wheel:
	$(PYTHON) -m pip wheel . -w build --verbose
python_sdist:
	$(PYTHON) -m build --sdist
test_install:
	python3 -m pip install dist/pocket_numpy-*.tar.gz --force-reinstall
.PHONY: build python_install python_wheel python_sdist test_install

test:
	build/pocketpy.exe tests/test_numpy.py
