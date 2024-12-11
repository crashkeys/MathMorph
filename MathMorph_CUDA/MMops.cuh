//
// Created by Irene on 05/12/2024.
//
#include "opencv2/highgui.hpp"
#ifndef MMOPS2_CUH
#define MMOPS2_CUH


__global__ void erode(uchar* d_input, uchar* d_output, int width, int height);
__global__ void dilate(uchar* d_input, uchar* d_output, int width, int height);

#endif //MMOPS2_CUH
