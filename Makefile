BIN_DIR = bin
SRC_DIR = src

CC = gcc
CFLAGS = -Wall -g
SPECIALFLAGS = -lnvidia-ml -lpthread

BIN = gpowerSAMPLER
SRC = power_measure.c

all:  $(BIN)

$(BIN):
	$(CC) $(CFLAGS) -I/usr/local/cuda/include  -I/usr/include/nvidia/gdk/ -L/usr/local/cuda/lib64 -lcudart -lcuda $(SPECIALFLAGS)  -o $(BIN_DIR)/$(BIN) $(SRC_DIR)/$(SRC)

clean:
	rm -f $(BIN_DIR)/*

install:
	cp $(BIN_DIR)/$(BIN) /usr/local/bin/$(BIN)

uninstall:
	rm -f /usr/local/bin/$(BIN)
