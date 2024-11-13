//
// Created by nenec on 05/11/2024.
//
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include "Erosion.h"

using namespace cv;
using namespace std;

int probe_size = 50;

int main( int argc, char** argv ) {
    setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    Mat frame = imread(R"(gsimage.jpg)",IMREAD_GRAYSCALE);
    Mat frameIn = imread(R"(gsimage.jpg)",IMREAD_GRAYSCALE);
    Mat element =cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(probe_size,probe_size));

    copyMakeBorder(frame, frame, probe_size, probe_size, probe_size, probe_size, 0);

    Mat erosionImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    Mat dilationImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    Mat closingImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
    Mat openingImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

    for (int i=0+probe_size; i<frame.rows-probe_size; i++) {
        for (int j=0+probe_size; j<frame.cols-probe_size; j++) {
            int erosion = erode(frame, element, i, j);
            int dilation = dilate(frame, element, i, j);
            erosionImg.at<uchar>(i,j) = erosion;
            dilationImg.at<uchar>(i,j) = dilation;
        }
    }

    for (int i=0+probe_size; i<closingImg.rows-probe_size; i++) {
        for (int j=0+probe_size; j<closingImg.cols-probe_size; j++) {
            int opening = dilate(erosionImg, element, i, j);
            int closing = erode(dilationImg, element, i, j);

            closingImg.at<uchar>(i,j) = closing;
            openingImg.at<uchar>(i,j) = opening;
        }
    }



    namedWindow("Source", WINDOW_AUTOSIZE);
    imshow("Source", frameIn); //original size


    namedWindow("Erosion", WINDOW_AUTOSIZE);
    imshow("Erosion", erosionImg);
    namedWindow("Dilation", WINDOW_AUTOSIZE);
    imshow("Dilation", dilationImg);

    namedWindow("Opening", WINDOW_AUTOSIZE);
    imshow("Opening", openingImg);
    namedWindow("Closing", WINDOW_AUTOSIZE);
    imshow("Closing", closingImg);

    waitKey(0);

    return 0;

}