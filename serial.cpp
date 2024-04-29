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

static __uint8_t convert(const float num) {
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
static void packing(float *matrix, __uint8_t *res, const unsigned int size) {
    int num_bytes = size / 4;
    for (unsigned int i = 0; i < size; i += 4) {
      __uint8_t currRes = 0;
      currRes |= convert(matrix[i]) << 6; 
      currRes |= convert(matrix[i+1]) << 4; 
      currRes |= convert(matrix[i+2]) << 2;
      currRes |= convert(matrix[i+3]);    
      res[i / 4] = currRes;
    }
}

// ex input shape:[100,b] weight_shape: [25,25] -> [100,b]
static void matmul_2b(float *input, __uint8_t *weight, float *output, size_t input_rows, size_t input_cols, size_t weight_rows, size_t weight_cols) {
  for(int i = 0; i < weight_rows; i++){
    for(int j = 0; j < input_cols; j++){
        float acc = 0;
        for(int k = 0; k < input_rows; k+=4){
        __uint8_t w = weight[i*weight_cols + k/4];
        //Unpack w into 4 elements
        __uint8_t w1 = (w >> 6) & 0x3;
        __uint8_t w2 = (w >> 4) & 0x3;
        __uint8_t w3 = (w >> 2) & 0x3;
        __uint8_t w4 = w & 0x3;

        printf("w1: %d, w2: %d, w3: %d, w4: %d\n", w1, w2, w3, w4);

        if (w1 == 0){
            acc -= input[j*input_rows + k];
        } else if (w1 == 2){
            acc += input[j*input_rows + k];
        }

        if (w2 == 0){
            acc -= input[j*input_rows + k+1];
        } else if (w2 == 2){
            acc += input[j*input_rows + k+1];
        }

        if (w3 == 0){
            acc -= input[j*input_rows + k+2];
        } else if (w3 == 2){
            acc += input[j*input_rows + k+2];
        }

        if (w4 == 0){
            acc -= input[j*input_rows + k+3];
        } else if (w4 == 2){
            acc += input[j*input_rows + k+3];
        }

        //Take the k,k+1,k+2,k+3 elements from input and dot product using if statements
        //update accumulator
        }
    output[i*input_cols + j] = acc;
    }
  }
}