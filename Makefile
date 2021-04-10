.PHONY: all clean

.DEFAULT: all


CXX = c++

CXXFLAGS = -ansi -pedantic -std=c++14  -O3

TARGETS = main

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : %: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
