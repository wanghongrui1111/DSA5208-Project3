# Makefile -- scaling_2d3d/src (MPI + OpenMP CG for the 2D / 3D Poisson problem)
#
#   make            serial + poisson into ../bin/
#   make serial     serial CG only (no MPI needed:  make serial CXX=g++)
#   make poisson    MPI + OpenMP driver (mpicxx)
# Overrides:  make MPICXX=mpicxx OMPFLAG=-fopenmp ARCH_FLAGS=-march=native BIN=../bin
#   macOS/Apple clang: OMPFLAG="-Xpreprocessor -fopenmp" + libomp.   NVHPC (nvc++): OMPFLAG=-mp
MPICXX     ?= mpicxx
CXX        ?= g++
OMPFLAG    ?= -fopenmp
ARCH_FLAGS ?=
CXXFLAGS   ?= -O3 -std=c++17 -Wall $(ARCH_FLAGS)
BIN        ?= ../bin

HDR := $(wildcard *.hpp)

.PHONY: all serial poisson clean
all: serial poisson
serial: $(BIN)/serial
poisson: $(BIN)/poisson

$(BIN):
	mkdir -p $(BIN)

# serial reference: plain C++, no MPI, no OpenMP
$(BIN)/serial: serial_main.cpp serial.cpp $(HDR) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ serial_main.cpp serial.cpp

$(BIN)/poisson: mpi_main.cpp solver.cpp serial.cpp $(HDR) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAG) -o $@ mpi_main.cpp solver.cpp serial.cpp

clean:
	rm -rf $(BIN)
