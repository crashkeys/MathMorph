#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <fstream>
#include <cuda_runtime_api.h>
#include "MMops.cuh"


using namespace  std;
using namespace cv;
#define ELEM_SIZE 15
__constant__ uchar ELEM[ELEM_SIZE*ELEM_SIZE];

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)


static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) {
        return;
    }

    std::cerr << statement << " returned " << cudaGetErrorString(err)
              << " (" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

int main (int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    Mat inputMat = imread("./images/arcane.jpg", cv::IMREAD_GRAYSCALE);
    Mat element = cv::getStructuringElement(cv::MORPH_CROSS, Size(ELEM_SIZE, ELEM_SIZE));

    uchar hostElem[ELEM_SIZE*ELEM_SIZE];
    std::memcpy(hostElem, element.data, ELEM_SIZE*ELEM_SIZE * sizeof(uchar));

    ofstream csvFile("C:\\Users\\Irene\\Desktop\\graphs\\ES2\\arcane.csv");
    csvFile << "size , time" << endl;

    int width = inputMat.cols;
    int height = inputMat.rows;

    int sizes[4] = {4, 8, 16, 32};
    for (int i = 0; i < 4; i++) {
        int reps = 0;
        while (reps < 50) {

            auto start = std::chrono::system_clock::now();

            uchar* d_input;
            uchar* d_output_EROSION;
            uchar* d_output_DILATION;
            uchar* d_output_OPENING;
            uchar* d_output_CLOSING;
            size_t size = width * height * sizeof(uchar);

            CUDA_CHECK_RETURN(cudaMalloc((void**)&d_input, size));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output_EROSION, size));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output_DILATION, size));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output_OPENING, size));
            CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output_CLOSING, size));

            CUDA_CHECK_RETURN(cudaMemcpy(d_input, inputMat.data, size, cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(ELEM, hostElem, sizeof(uchar)*ELEM_SIZE*ELEM_SIZE, 0));

            int blockSize = sizes[i];
            dim3 dimBlock(blockSize, blockSize);
            dim3 dimGrid((width + blockSize - 1)/blockSize , (height + blockSize - 1)/blockSize);

            /*EROSION AND DILATION*/
            erode<<<dimGrid, dimBlock>>>(d_input, d_output_EROSION, width, height);
            dilate<<<dimGrid, dimBlock>>>(d_input, d_output_DILATION, width, height);
            cudaDeviceSynchronize();

            /*OPENING AND CLOSING*/
            erode<<<dimGrid, dimBlock>>>(d_output_DILATION, d_output_CLOSING, width, height);
            dilate<<<dimGrid, dimBlock>>>(d_output_EROSION, d_output_OPENING, width, height);
            cudaDeviceSynchronize();

            //gpu -> cpu for visualization
            Mat output_EROSION(height, width, CV_8UC1); //type=CV_32SC1 se lo voglio fare a int
            Mat output_DILATION(height, width, CV_8UC1);
            Mat output_OPENING(height, width, CV_8UC1);
            Mat output_CLOSING(height, width, CV_8UC1);
            cudaMemcpy(output_EROSION.data, d_output_EROSION, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_DILATION.data, d_output_DILATION, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_OPENING.data, d_output_OPENING, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_CLOSING.data, d_output_CLOSING, size, cudaMemcpyDeviceToHost);

            auto end = std::chrono::system_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            csvFile << sizes[i] << "," << time.count() << endl;

            /*cout << "Example is using: \n \t image size: " << width << "x" << height << "\n \t probe type: MORPH_ELLIPSE" << endl;
            cout << "Time to perform Erosion, Dilation, Opening and Closing: " << time.count() << " microsec." << endl;
            cout << "\n\n" << endl;*/

            cudaFree(d_input);
            cudaFree(d_output_EROSION);
            cudaFree(d_output_DILATION);
            cudaFree(d_output_OPENING);
            cudaFree(d_output_CLOSING);
            reps++;
        }
    }

    /*//show
    imshow("Erosion Result", output_EROSION);
    imshow("Dilation Result", output_DILATION);
    imshow("Opening Result", output_OPENING);
    imshow("Closing Result", output_CLOSING);

    waitKey(0);*/


    return 0;
}

