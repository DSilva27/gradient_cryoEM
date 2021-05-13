IDIR = include
CC = g++
CFLAGS = -std=c++14 -O3 -I$(IDIR) -lm -lfftw3f 

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
