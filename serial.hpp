#include <cstdio>
#include <cmath>

using namespace std;

#define EPS 1e-5

static void quantize2bit(float *matrix, const unsigned int num_elements) {
    // Calculate the gamma value (mean absolute value of the weights)
    float gamma = 0.0f;
    for (unsigned int i = 0; i < num_elements; ++i) {
        gamma += fabs(matrix[i]);
    }
    gamma /= num_elements;
    gamma += EPS; // add epsilon (assuming eps is a small positive value)

    // Quantize the weights
    for (unsigned int i = 0; i < num_elements; ++i) {
        float scaled = matrix[i] / gamma;
        float abs_scaled = fabs(scaled);
        float rounded = round(abs_scaled);
        rounded = fmin(rounded, 1.0f); // clamp to max 1.0
        matrix[i] = (scaled > 0)? rounded : -rounded;
    }
}

__int8_t convert(const float num) {
    return (num == -1) ? 0 : (num == 0) ? 1 : 2;
}

/*
Take an array of floats that are one of {-1, 0, 1} and pack each group of 4 into
an 8bit int:
-1 -> 00
0 -> 01
1 -> 10
[-1,0,1,1] -> [0b00011010]
*/
void packing(float *matrix, __int8_t *res, const unsigned int size) {
        int num_bytes = size / 4;
        for (unsigned int i = 0; i < size; i += 4) {
            __int8_t currRes = 0;
            currRes |= convert(matrix[i]) << 6; 
            currRes |= convert(matrix[i+1]) << 4; 
            currRes |= convert(matrix[i+2]) << 2;
            currRes |= convert(matrix[i+3]);    
            res[i / 4] = currRes;
        }
}

// input shape:[100,100] weight_shape: [25,25]
void matmul_2b(float *input, __int8_t weight) {
}