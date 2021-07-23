IDIR = include
CC = g++
CFLAGS = -g -std=c++17 -O3 -I$(IDIR) -lm -lfftw3f 

SDIR = src

_DEPS = gradcv.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_SRC = main.cpp gradcv.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

gradcv.out: $(SRC) $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm gradcv.out
