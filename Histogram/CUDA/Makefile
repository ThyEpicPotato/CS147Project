NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_61  
EXE = histogram
SRC = src/main.cu

all: $(EXE)

$(EXE):
	$(NVCC) $(SRC) -o $(EXE) $(NVCC_FLAGS)

clean:
	rm -f $(EXE)

