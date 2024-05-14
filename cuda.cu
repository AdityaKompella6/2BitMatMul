#define EPS 1e-5
#include <cstdio>
#include <cmath>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <curand.h>
#include <random>
#include <chrono>
#include <iostream>
//Computes W @ C.T
__global__ void quantize2bitKernel(float *matrix, unsigned int num_elements)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements)
  {
    __shared__ float gamma_shared;
    if (threadIdx.x == 0)
      gamma_shared = 0.0f;
    __syncthreads();
    gamma_shared += fabs(matrix[idx]);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      gamma_shared /= num_elements;
      gamma_shared += EPS;
    }
    __syncthreads();
    float scaled = matrix[idx] / gamma_shared;
    float abs_scaled = fabs(scaled);
    float rounded = round(abs_scaled);
    rounded = fmin(rounded, 1.0f);
    matrix[idx] = (scaled > 0) ? rounded : -rounded;
  }
}

__device__ __uint8_t convert(const float num)
{
  return (num == -1) ? 0 : (num == 0) ? 1
                                      : 2;
}

__global__ void packingKernel(float *matrix, __uint8_t *res, unsigned int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    __uint8_t currRes = 0;
    currRes |= convert(matrix[idx]) << 6;
    currRes |= convert(matrix[idx + 1]) << 4;
    currRes |= convert(matrix[idx + 2]) << 2;
    currRes |= convert(matrix[idx + 3]);
    res[idx / 4] = currRes;
  }
}

__global__ void matmul_2bKernel(float *input, __uint8_t *weight, float *output,
                                size_t input_rows, size_t input_cols, size_t weight_rows, size_t weight_cols)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < weight_rows * input_cols)
  {
    int i = idx / input_cols;
    int j = idx % input_cols;
    float acc = 0;
    for (int k = 0; k < input_rows; k += 4)
    {
      __uint8_t w = weight[i * weight_cols + k / 4];
      __uint8_t w1 = (w >> 6) & 0x3;
      __uint8_t w2 = (w >> 4) & 0x3;
      __uint8_t w3 = (w >> 2) & 0x3;
      __uint8_t w4 = w & 0x3;

      if (w1 == 0)
      {
        acc -= input[j * input_rows + k];
      }
      else if (w1 == 2)
      {
        acc += input[j * input_rows + k];
      }

      if (w2 == 0)
      {
        acc -= input[j * input_rows + k + 1];
      }
      else if (w2 == 2)
      {
        acc += input[j * input_rows + k + 1];
      }

      if (w3 == 0)
      {
        acc -= input[j * input_rows + k + 2];
      }
      else if (w3 == 2)
      {
        acc += input[j * input_rows + k + 2];
      }

      if (w4 == 0)
      {
        acc -= input[j * input_rows + k + 3];
      }
      else if (w4 == 2)
      {
        acc += input[j * input_rows + k + 3];
      }
    }
    output[i * input_cols + j] = acc;
  }
}

int main(int argc, char *argv[])
{
  int numRuns = 100;
  unsigned int lda = atoi(argv[1]);
  unsigned int n = lda * lda;

  float *h_A, *h_C, *h_output;
  h_A = new float[n];
  h_C = new float[n];
  h_output = new float[n];

  // Randomly generate values for A and C matrices
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0, 10.0);

  for (unsigned int i = 0; i < n; ++i)
  {
    h_A[i] = dis(gen);
    h_C[i] = dis(gen);
  }

  float *d_A, *d_C, *d_output;
  cudaMalloc((void **)&d_A, sizeof(float) * n);
  cudaMalloc((void **)&d_C, sizeof(float) * n);
  cudaMalloc((void **)&d_output, sizeof(float) * n);

  cudaMemcpy(d_A, h_A, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, sizeof(float) * n, cudaMemcpyHostToDevice);
  double totalTime = 0.0;
  for (int i = 0; i < numRuns; i++)
  {
    auto start = std::chrono::steady_clock::now();
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    quantize2bitKernel<<<gridSize, blockSize>>>(d_A, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    totalTime += std::chrono::duration<double, std::milli>(diff).count();
  }
  double avgTime = totalTime / numRuns;
  std::cout << "quantize2bitKernel execution time: " << avgTime << " ms" << std::endl;

  // std::cout << "Quantized weight matrix:" << std::endl;
  // float *h_quantized_A;
  // h_quantized_A = new float[n];
  // cudaMemcpy(h_quantized_A, d_A, sizeof(float) * n, cudaMemcpyDeviceToHost);
  // for (unsigned int i = 0; i < n; ++i) {
  //     std::cout << h_quantized_A[i] << " ";
  //     if ((i + 1) % lda == 0) {
  //         std::cout << std::endl;
  //     }
  // }
  // delete[] h_quantized_A;

  // // Print C matrix
  // std::cout << "C matrix:" << std::endl;
  // for (unsigned int i = 0; i < n; ++i) {
  //     std::cout << h_C[i] << " ";
  //     if ((i + 1) % lda == 0) {
  //         std::cout << std::endl;
  //     }
  // }

  __uint8_t *h_packed;
  h_packed = new __uint8_t[n / 4];
  __uint8_t *d_packed;
  cudaMalloc((void **)&d_packed, sizeof(__uint8_t) * (n / 4));
  int blockSize = 256;
  totalTime = 0.0;
  for (int i = 0; i < numRuns; i++)
  {
    auto start = std::chrono::steady_clock::now();
    int gridSize = (n + blockSize - 1) / blockSize;
    packingKernel<<<gridSize, blockSize>>>(d_A, d_packed, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    totalTime += std::chrono::duration<double, std::milli>(diff).count();
  }

  avgTime = totalTime / numRuns;
  std::cout << "packingKernel execution time: " << avgTime << " ms" << std::endl;

  int rows = lda;
  int cols = lda;
  totalTime = 0.0;
  blockSize = 256;
  for (int i = 0; i < numRuns; i++)
  {
    auto start = std::chrono::steady_clock::now();
    int gridSize = (rows * cols + blockSize - 1) / blockSize;
    matmul_2bKernel<<<gridSize, blockSize>>>(d_C, d_packed, d_output, rows, cols, rows, cols / 4);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    totalTime += std::chrono::duration<double, std::milli>(diff).count();
  }

  avgTime = totalTime / numRuns;
  std::cout << "Average matmul_2bKernel execution time: " << avgTime << " ms" << std::endl;

  // cudaMemcpy(h_output, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);
  // std::cout << "Output matrix:" << std::endl;
  // cudaMemcpy(h_output, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);
  // for (unsigned int i = 0; i < n; ++i)
  // {
  //   std::cout << h_output[i] << " ";
  //   if ((i + 1) % lda == 0)
  //   {
  //     std::cout << std::endl;
  //   }
  // }

  delete[] h_A;
  delete[] h_C;
  delete[] h_output;
  delete[] h_packed;
  cudaFree(d_A);
  cudaFree(d_C);
  cudaFree(d_output);
  cudaFree(d_packed);

  return 0;
}
