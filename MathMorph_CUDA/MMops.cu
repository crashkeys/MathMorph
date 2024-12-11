#include "opencv2/highgui.hpp"
#include "MMops.cuh"

#define ELEM_SIZE 15
extern __constant__ uchar ELEM[ELEM_SIZE*ELEM_SIZE];


__global__ void erode(uchar* d_input, uchar* d_output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        uchar value = 255;

        for (int el_id = 0; el_id < ELEM_SIZE*ELEM_SIZE; el_id++) {
            int r = floorf(el_id/ELEM_SIZE);
            int c = el_id % ELEM_SIZE;

            int newRow = row + r;
            int newCol = col + c;

            if (newRow>=0 && newRow<height && newCol>=0 && newCol<width) {
                if (ELEM[el_id] == 1) {
                    uchar temp = d_input[newRow*width + newCol];
                    value = value < temp ? value : temp;  //MIN
                }
            }
        }
        d_output[row * width + col] = value;
    }
}


__global__ void dilate(uchar* d_input, uchar* d_output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        uchar value = 0;

        for (int el_id = 0; el_id < ELEM_SIZE*ELEM_SIZE; el_id++) {
            int r = floorf(el_id/ELEM_SIZE);
            int c = el_id % ELEM_SIZE;

            int newRow = row - r;
            int newCol = col - c;

            if (newRow>=0 && newRow<height && newCol>=0 && newCol<width) {
                if (ELEM[el_id] == 1) {
                    uchar temp = d_input[newRow*width + newCol];
                    value = value > temp ? value : temp;  //MAX
                }
            }
        }
        d_output[row * width + col] = value;
    }
}