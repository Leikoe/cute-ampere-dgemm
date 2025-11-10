CFLAGS=-I/usr/local/cuda/include -Icutlass/include -Icutlass/tools/util/include -arch=sm_80 --std=c++17 --expt-relaxed-constexpr -ftemplate-backtrace-limit=0

SRC_FOLDER=./src

baseline: $(SRC_FOLDER)/baseline.cu
	nvcc $(SRC_FOLDER)/baseline.cu -o baseline $(CFLAGS)

cublas: $(SRC_FOLDER)/cublas.cu
	nvcc $(SRC_FOLDER)/cublas.cu -o cublas_dgemm $(CFLAGS) -lcublas

baseline_m8n8k8: $(SRC_FOLDER)/baseline_m8n8k8.cu
	nvcc $(SRC_FOLDER)/baseline_m8n8k8.cu -o baseline_m8n8k8 $(CFLAGS)

baseline_m16n16k16: $(SRC_FOLDER)/baseline_m16n16k16.cu
	nvcc $(SRC_FOLDER)/baseline_m16n16k16.cu -o baseline_m16n16k16 $(CFLAGS)

baseline_m16n16k16_4W: $(SRC_FOLDER)/baseline_m16n16k16_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m16n16k16_4W.cu -o baseline_m16n16k16_4W $(CFLAGS)

baseline_m32n32k32: $(SRC_FOLDER)/baseline_m32n32k32.cu
	nvcc $(SRC_FOLDER)/baseline_m32n32k32.cu -o baseline_m32n32k32 $(CFLAGS)

baseline_m32n32k32_4W: $(SRC_FOLDER)/baseline_m32n32k32_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m32n32k32_4W.cu -o baseline_m32n32k32_4W $(CFLAGS)

baseline_m64n64k64: $(SRC_FOLDER)/baseline_m64n64k64.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k64.cu -o baseline_m64n64k64 $(CFLAGS)

baseline_m64n64k64_4W: $(SRC_FOLDER)/baseline_m64n64k64_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k64_4W.cu -o baseline_m64n64k64_4W $(CFLAGS)

pipelined: $(SRC_FOLDER)/pipelined.cu
	nvcc $(SRC_FOLDER)/pipelined.cu -o pipelined $(CFLAGS)

clean:
	rm baseline baseline_m8n8k8 baseline_m16n16k16 baseline_m16n16k16_4W baseline_m32n32k32 pipelined
	rm viz.tex viz.log viz.aux viz.pdf
