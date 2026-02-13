all:
	@echo nothing special
.PHONY: all

clean:
	rm -rf build
.PHONY: clean

build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --config Release
.PHONY: build

test:
	build/test_numpy tests/test_numpy.py
