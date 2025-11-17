CFLAGS=-I/usr/local/cuda/include -Icutlass/include -Icutlass/tools/util/include -arch=sm_80 --std=c++17 --expt-relaxed-constexpr -ftemplate-backtrace-limit=0

SRC_FOLDER=./src

all: baseline cublas baseline_m8n8k8 baseline_m8n8k8_128b baseline_m16n16k16_128b baseline_m16n16k16_128b_4W baseline_m32n32k32_128b baseline_m32n32k32_128b_4W baseline_m64n64k32_128b baseline_m64n64k32_128b_4W baseline_m64n64k64_128b baseline_m64n64k64_128b_4W baseline_m128n64k16_128b_4W baseline_m128n128k16_128b_4W pipelined

baseline: $(SRC_FOLDER)/baseline.cu
	nvcc $(SRC_FOLDER)/baseline.cu -o baseline $(CFLAGS)

cublas: $(SRC_FOLDER)/cublas.cu
	nvcc $(SRC_FOLDER)/cublas.cu -o cublas_dgemm $(CFLAGS) -lcublas

baseline_m8n8k8: $(SRC_FOLDER)/baseline_m8n8k8.cu
	nvcc $(SRC_FOLDER)/baseline_m8n8k8.cu -o baseline_m8n8k8 $(CFLAGS)

baseline_m8n8k8_128b: $(SRC_FOLDER)/baseline_m8n8k8_128b.cu
	nvcc $(SRC_FOLDER)/baseline_m8n8k8_128b.cu -o baseline_m8n8k8_128b $(CFLAGS)

baseline_m16n16k16_128b: $(SRC_FOLDER)/baseline_m16n16k16_128b.cu
	nvcc $(SRC_FOLDER)/baseline_m16n16k16_128b.cu -o baseline_m16n16k16_128b $(CFLAGS)

baseline_m16n16k16_128b_4W: $(SRC_FOLDER)/baseline_m16n16k16_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m16n16k16_128b_4W.cu -o baseline_m16n16k16_128b_4W $(CFLAGS)

baseline_m32n32k32_128b: $(SRC_FOLDER)/baseline_m32n32k32_128b.cu
	nvcc $(SRC_FOLDER)/baseline_m32n32k32_128b.cu -o baseline_m32n32k32_128b $(CFLAGS)

baseline_m32n32k32_128b_4W: $(SRC_FOLDER)/baseline_m32n32k32_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m32n32k32_128b_4W.cu -o baseline_m32n32k32_128b_4W $(CFLAGS)

baseline_m64n64k32_128b: $(SRC_FOLDER)/baseline_m64n64k32_128b.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k32_128b.cu -o baseline_m64n64k32_128b $(CFLAGS)

baseline_m64n64k32_128b_4W: $(SRC_FOLDER)/baseline_m64n64k32_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k32_128b_4W.cu -o baseline_m64n64k32_128b_4W $(CFLAGS)

baseline_m64n64k64_128b: $(SRC_FOLDER)/baseline_m64n64k64_128b.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k64_128b.cu -o baseline_m64n64k64_128b $(CFLAGS)

baseline_m64n64k64_128b_4W: $(SRC_FOLDER)/baseline_m64n64k64_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m64n64k64_128b_4W.cu -o baseline_m64n64k64_128b_4W $(CFLAGS)

baseline_m128n64k16_128b_4W: $(SRC_FOLDER)/baseline_m128n64k16_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m128n64k16_128b_4W.cu -o baseline_m128n64k16_128b_4W $(CFLAGS)

baseline_m128n128k16_128b_4W: $(SRC_FOLDER)/baseline_m128n128k16_128b_4W.cu
	nvcc $(SRC_FOLDER)/baseline_m128n128k16_128b_4W.cu -o baseline_m128n128k16_128b_4W $(CFLAGS)

pipelined: $(SRC_FOLDER)/pipelined.cu
	nvcc $(SRC_FOLDER)/pipelined.cu -o pipelined $(CFLAGS)

clean:
	rm baseline cublas baseline_m8n8k8 baseline_m8n8k8_128b baseline_m16n16k16_128b baseline_m16n16k16_128b_4W baseline_m32n32k32_128b baseline_m32n32k32_128b_4W baseline_m64n64k32_128b baseline_m64n64k32_128b_4W baseline_m64n64k64_128b baseline_m64n64k64_128b_4W baseline_m128n64k16_128b_4W baseline_m128n128k16_128b_4W pipelined
	rm viz.tex viz.log viz.aux viz.pdf
