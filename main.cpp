//
// Created by nenec on 05/11/2024.
//
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include "MMops.h"

using namespace cv;
using namespace std;

int probe_size = 15;

int main( int argc, char** argv ) {
    setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    //String img_name = "gsimage.jpg";
    Mat frame = imread("./images/cityanime.jpg", IMREAD_GRAYSCALE);
    Mat frameIn = imread("./images/cityanime.jpg", IMREAD_GRAYSCALE);
    Mat element =cv::getStructuringElement(MORPH_ELLIPSE, Size(probe_size,probe_size));
    copyMakeBorder(frame, frame, probe_size, probe_size, probe_size, probe_size, 0);

    ofstream csvFile("C:\\Users\\Irene\\Desktop\\graphs\\ES2\\sequential\\arcane.csv");
    csvFile << "size" << "time" << endl;

    int times = 0;
    while (times < 10) {

        auto start = std::chrono::system_clock::now();

        Mat erosionImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
        Mat dilationImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
        Mat closingImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
        Mat openingImg = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));


        /*EROSION & DILATION*/
        for (int i=0+probe_size; i<frame.rows-probe_size; i++) {
            for (int j=0+probe_size; j<frame.cols-probe_size; j++) {
                int erosion = erode(frame, element, i, j);
                int dilation = dilate(frame, element, i, j);
                erosionImg.at<uchar>(i,j) = erosion;
                dilationImg.at<uchar>(i,j) = dilation;
            }
        }

        /*OPENING & CLOSING*/
        for (int i=0+probe_size; i<closingImg.rows-probe_size; i++) {
            for (int j=0+probe_size; j<closingImg.cols-probe_size; j++) {
                int opening = dilate(erosionImg, element, i, j);
                int closing = erode(dilationImg, element, i, j);

                closingImg.at<uchar>(i,j) = closing;
                openingImg.at<uchar>(i,j) = opening;
            }
        }

        auto end = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        csvFile << frameIn.rows << "," << time.count() << endl;
        times++;
    }

    //cout << "Time to perform Erosion, Dilation, Opening and Closing: " << time.count() << " millisec." << endl;

    /*imshow("Source", frameIn); //original
    imshow("Erosion", erosionImg);
    imshow("Dilation", dilationImg);
    imshow("Opening", openingImg);
    imshow("Closing", closingImg);

    waitKey();*/

    return 0;

}