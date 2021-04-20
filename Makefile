.PHONY: all clean

.DEFAULT: all


CXX = g++

CXXFLAGS = -std=c++14 -O3 -openmp

SDIR = src
_SRC = main.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

TARGETS = main.out

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : $(SRC)
			 $(CXX) $(CXXFLAGS) -o $@ $<

