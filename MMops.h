//
// Created by nenec on 05/11/2024.
//
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

#ifndef EROSION_H
#define EROSION_H
int erode(Mat& frame, Mat& element, int i, int j);
int dilate(Mat& frame, Mat& element, int i, int j);
#endif //EROSION_H
