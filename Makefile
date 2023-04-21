build:
	cmake -S . -B ./build
.PHONY: build

test:
	cd ./build && make && cd ./test && ./test
.PHONY: test