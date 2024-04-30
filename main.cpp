#include "serial.cpp"
#include <random>
int main(int argc, char* argv[]) { 
	// row-major matrix
	unsigned int n = atoi(argv[1]);

	float* A = new float[n];
    float* C = new float[n];
    float* output = new float[n];

    // Randomly generate values for A and C matrices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0, 10.0);

    for (unsigned int i = 0; i < n; ++i) {
        A[i] = dis(gen);
        C[i] = dis(gen);
    }

	quantize2bit(A, n);

	for (unsigned int r = 0; r < 4; ++r) {
		for (unsigned int c = 0; c < 4; ++c) {
			std::printf("%f ", A[c + r*4]);
		}
    	std::printf("\n");
	}


	__uint8_t packed[n/4];

	packing(A, packed, n);

	int rows = std::sqrt(n);
	int cols = std::sqrt(n);

	matmul_2b(C, packed, output, rows, cols, rows, cols/4);

	for (unsigned int r = 0; r < rows; ++r) {
		for (unsigned int c = 0; c < cols; ++c) {
			std::printf("%f ", output[c + r*4]);
		}
    	std::printf("\n");
	}

	return 0;
 }
 //input = [-0.15,0.1,0.6,0.4]
 //w = [-1,0,1,0]
 //acc = 0
 //for i in range(4)
 //if w[i] == -1 then acc -= input[i]
 //if w[i] == 1 then acc += input[i]
