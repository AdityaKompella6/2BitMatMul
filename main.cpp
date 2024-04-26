#include "serial.cpp"

int main(int argc, char* argv[]) { 
	// row-major matrix
	const unsigned int n = 16;
	float A[n] = {
		6.71309, 1.84700, 2.61925, 7.53522,
		8.16581, 8.69585, 8.15835, 7.02395,
		-8.97898, -8.22587, -5.07179, -2.09032,
		-7.82172, -8.09718, -2.40825, -8.89823
	};

	quantize2bit(A, n);

	for (unsigned int r = 0; r < 4; ++r) {
		for (unsigned int c = 0; c < 4; ++c) {
			std::printf("%f ", A[c + r*4]);
		}
    	std::printf("\n");
	}

	__uint8_t packed[4];

	packing(A, packed, n);

	for (unsigned int r = 0; r < 1; ++r) {
		for (unsigned int c = 0; c < 4; ++c) {
			std::printf("%u ", packed[c + r*4]);
		}
    	std::printf("\n");
	}

	float output[16];

	matmul_2b(A, packed, output, 4, 4, 4, 1);

	for (unsigned int r = 0; r < 4; ++r) {
		for (unsigned int c = 0; c < 4; ++c) {
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
