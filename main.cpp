#include "serial.cpp"
#include <random>
#include <chrono>
#include <iostream>
int main(int argc, char* argv[]) { 
	// row-major matrix
	unsigned int lda = atoi(argv[1]);
	unsigned int n = lda*lda;

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
	
	auto start = std::chrono::steady_clock::now();
	quantize2bit(A, n);
	auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << "quantize2bit execution time: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

	// for (unsigned int r = 0; r < 4; ++r) {
	// 	for (unsigned int c = 0; c < 4; ++c) {
	// 		std::printf("%f ", A[c + r*4]);
	// 	}
    // 	std::printf("\n");
	// }


	__uint8_t packed[n/4];
	start = std::chrono::steady_clock::now();
	packing(A, packed, n);
	end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "packing execution time: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

	int rows = lda;
	int cols = lda;

	start = std::chrono::steady_clock::now();
	matmul_2b(C, packed, output, rows, cols, rows, cols/4);
	end = std::chrono::steady_clock::now();
    diff = end - start;
    std::cout << "matmul_2b execution time: " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

	// for (unsigned int r = 0; r < rows; ++r) {
	// 	for (unsigned int c = 0; c < cols; ++c) {
	// 		std::printf("%f ", output[c + r*4]);
	// 	}
    // 	std::printf("\n");
	// }

	return 0;
 }
 //input = [-0.15,0.1,0.6,0.4]
 //w = [-1,0,1,0]
 //acc = 0
 //for i in range(4)
 //if w[i] == -1 then acc -= input[i]
 //if w[i] == 1 then acc += input[i]
