IDIR = include
CC = g++
CFLAGS = -std=c++17 -O3 -g -I$(IDIR) -lm -lfftw3 -qopt-report

SDIR = src

_DEPS = gradcv.h 
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_SRC = main.cpp gradcv.cpp
SRC = $(patsubst %,$(SDIR)/%,$(_SRC))

gradcv.out: $(SRC) $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
