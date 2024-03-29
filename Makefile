IDIR = include
CC = mpic++
CFLAGS = -std=c++17 -O1 -fopenmp -g -I$(IDIR) -lm -lfftw3 

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
