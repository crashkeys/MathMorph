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


__global__ void MM_ops_shared(uchar* d_input, uchar* d_output_first, uchar* d_output_second, int width, int height, bool opening) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;
    extern __shared__ uchar sharedMem[];

    int pdd = ELEM_SIZE / 2;
    int paddedSize = blockDim.x + 2*pdd;

    int sharedRow = ty + pdd;
    int sharedCol = tx + pdd;

    //Central pixel
    if (row < height && col < width) {
        sharedMem[sharedRow*paddedSize + sharedCol] = d_input[row*width + col];
    }
    __syncthreads();

    // Load padding pixels
    if (tx < pdd) { // Left
        int leftCol = col - pdd;
        sharedMem[sharedRow*paddedSize + (sharedCol - pdd)] = d_input[row * width + leftCol];
    }
    if (tx >= blockDim.x - pdd) { // Right
        int rightCol = col + pdd;
        sharedMem[sharedRow*paddedSize + (sharedCol + pdd)] = d_input[row * width + rightCol];
    }
    if (ty < pdd) { // Top
        int topRow = row - pdd;
        sharedMem[(sharedRow - pdd) * paddedSize + sharedCol] = d_input[topRow * width + col];
    }
    if (ty >= blockDim.y - pdd) { // Bottom
        int bottomRow = row + pdd;
        sharedMem[(sharedRow + pdd) * paddedSize + sharedCol] = d_input[bottomRow * width + col];
    }

    // Load corner pixels
    if (tx < pdd && ty < pdd) { // Top-left
        int cornerRow = row - pdd;
        int cornerCol = col - pdd;
        sharedMem[(sharedRow - pdd) * paddedSize + (sharedCol - pdd)] = d_input[cornerRow * width + cornerCol];
    }
    if (tx < pdd && ty >= blockDim.y - pdd) { // Bottom-left
        int cornerRow = row + pdd;
        int cornerCol = col - pdd;
        sharedMem[(sharedRow + pdd) * paddedSize + (sharedCol - pdd)] = d_input[cornerRow * width + cornerCol];
    }
    if (tx >= blockDim.x - pdd && ty < pdd) { // Top-right
        int cornerRow = row - pdd;
        int cornerCol = col + pdd;
        sharedMem[(sharedRow - pdd) * paddedSize + (sharedCol + pdd)] = d_input[cornerRow * width + cornerCol];
    }
    if (tx >= blockDim.x - pdd && ty >= blockDim.y - pdd) { // Bottom-right
        int cornerRow = row + pdd;
        int cornerCol = col + pdd;
        sharedMem[(sharedRow + pdd) * paddedSize + (sharedCol + pdd)] = d_input[cornerRow * width + cornerCol];
    }
    __syncthreads();

    //Operations
    if (row < height && col < width) {
        uchar er_value = 255;
        uchar dil_value = 0;

        for (int el_id = 0; el_id < ELEM_SIZE * ELEM_SIZE; ++el_id) {
            int r = el_id / ELEM_SIZE - pdd;
            int c = el_id % ELEM_SIZE - pdd;

            if (ELEM[el_id] == 1) {
                uchar temp = sharedMem[(sharedRow + r) * paddedSize + (sharedCol + c)];
                er_value = min(er_value, temp);
                dil_value = max(dil_value, temp);
            }
        }

        //If second output, perform normal erosion and dilation
        if (d_output_second != nullptr) {
            d_output_first[row * width + col] = er_value;
            d_output_second[row * width + col] = dil_value;
        }
        else //Otherwise perfom normal erosion and dilation
        {
            if(opening==false)
                d_output_first[row * width + col] = er_value; //closing
            else
                d_output_first[row * width + col] = dil_value; //opening
        }
    }
}
