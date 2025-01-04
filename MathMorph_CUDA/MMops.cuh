//
// Created by Irene on 05/12/2024.
//
#include "opencv2/highgui.hpp"
#ifndef MMOPS2_CUH
#define MMOPS2_CUH


__global__ void erode(uchar* d_input, uchar* d_output, int width, int height);
__global__ void dilate(uchar* d_input, uchar* d_output, int width, int height);
__global__ void MM_ops_shared(uchar* d_input, uchar* d_output_first, uchar* d_output_second, int width, int height, bool opening=false);

#endif //MMOPS2_CUH
