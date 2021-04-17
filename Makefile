.PHONY: all clean

.DEFAULT: all


CXX = g++
#-fopenmp
CXXFLAGS = -std=c++14 -O2

TARGETS = main.out

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : %: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
