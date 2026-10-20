# header-only, this just builds the tests + examples
# eigen somewhere else? make test EIGEN=/opt/homebrew/include/eigen3
CXX ?= g++
EIGEN ?= $(patsubst -I%,%,$(firstword $(shell pkg-config --cflags-only-I eigen3 2>/dev/null) -I/usr/include/eigen3))
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
BUILD ?= build

FLAGS := $(CXXFLAGS) -Iinclude -isystem $(EIGEN)
HDR := include/cmaes/cmaes.hpp

all: $(BUILD)/test_cmaes $(BUILD)/rosenbrock $(BUILD)/ask_tell

$(BUILD)/test_cmaes: tests/test_cmaes.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(FLAGS) $< -o $@

$(BUILD)/%: examples/%.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(FLAGS) $< -o $@ -pthread

test: all
	$(BUILD)/test_cmaes

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
