//
// Created by nenec on 05/11/2024.
//

#include "MMops.h"

int erode(Mat& frame, Mat& element, int i, int j){
    int value = 255;
    for (int u=0; u<element.rows; u++) {
        for (int v=0; v<element.cols; v++) {
            if (element.at<uchar>(u,v) == 1) {
                int temp = frame.at<uchar>(i+u, j+v);
                value = min(temp, value);
            }
        }
    }
    return value;
}

int dilate(Mat& frame, Mat& element, int i, int j){
    int value = 0;
        for (int u=0; u<element.rows; u++) {
            for (int v=0; v<element.cols; v++) {
                if (element.at<uchar>(u,v) == 1) {
                    int temp = frame.at<uchar>(i-u, j-v);
                    value = max(temp, value);
                }
            }
        }
    return value;
}
