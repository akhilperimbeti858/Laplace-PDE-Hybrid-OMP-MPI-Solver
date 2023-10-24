#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include <sys/stat.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>


using namespace cv;
using namespace std;
#define BLOCK_SIZE 32

__global__ void RGB2GREY(uchar4 *image_RGB, unsigned char *image_GREY, int cols, int rows);

int main(int argc, char** argv)
{
    string image_name = "bear.jpg";
    string image_path = "/home/perimbeti/COMP605_HW4/bear.jpg";

    // Reading in the input image file ("bear.jpg")
    cv::Mat image = imread(image_path.c_str(), IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "[ERROR] Couldn't open file: " << image_path << endl;
              return 1;
    }

    cv::Mat image_RGB;
    cv::cvtColor(image, image_RGB, cv::COLOR_BGR2RGBA);

    uint num_cols = image.cols; // image width
    uint num_rows = image.rows; // image height

    cudaEvent_t start, end;
    float time_elapsed;

    int i = 0;  // used for kernel call repitions for time differences

    unsigned char *host_output;
    unsigned char *dim_output;
    uchar4 *dim_input;

    printf("\n");
    cout << "----------------------------------------------------------------" << endl;
    cout << " This program uses CUDA to convert an image in RGB to Greyscale" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "              Image Title: bear.jpg                     " << endl;
    cout << "   Image Dimenstions (rows x cols) = " << num_rows << " x " << num_cols << endl;
    cout << "----------------------------------------------------------------" << endl;

    int input_size = num_cols * num_rows * sizeof(uchar4);
    int output_size = num_cols * num_rows * sizeof(unsigned char);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(ceil(num_cols/(float)dimBlock.x) ,ceil(num_rows/(float)dimBlock.y));

    cout << "             GRID SIZE (Y x X) = " << dimGrid.y << " x "  << dimGrid.x <<endl;
    cout << "            BLOCK SIZE (N x N) = " << dimBlock.x << " x " << dimBlock.y <<endl;
    cout << "----------------------------------------------------------------" << endl;

    // Starting the Timer
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    // Memory Allocation for RGB Input and Output Image GPU
    cudaMalloc((void**)&dim_input, input_size);
    cudaMalloc((void**)&dim_output, output_size);

    // Memory Allocation for output greyscale image on CPU
    host_output = (unsigned char *)malloc(output_size);

    uchar4 *host_input = (uchar4 *)image_RGB.ptr<unsigned char>(0); // Conversion to 1D array

    cudaMemcpy(dim_input, host_input, input_size, cudaMemcpyHostToDevice); // Copies the input image to the device

    // CALL to Conversion KERNEL - looped 1000 times for execution time differences

    for(i = 0; i < 1000; i++) {
        RGB2GREY<<<dimGrid,dimBlock>>>(dim_input, dim_output, num_cols, num_rows);
    }

    // Copy output image results to CPU
    cudaMemcpy(host_output, dim_output,output_size, cudaMemcpyDeviceToHost);

    /*  --- STOP THE TIMER  -- */
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_elapsed, start, end);

    cout << "                   Time Elapsed: " << time_elapsed << " ms  "  << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "            Output Image Title: greyscale_bear.jpg     " << endl;
    cout << "----------------------------------------------------------------" << endl;
    printf("\n");

    // Writing greyscale image to output file (""greyscale_bear.jpg)
    cv::Mat output_image = Mat(num_rows, num_cols, CV_8UC1, host_output);
    imwrite("greyscale_"+image_name, output_image);

    //Free all allocated memory
    cudaFree(dim_input);
    cudaFree(dim_output);
    free(host_output);

    return 0;
}

// Kernel call for Image RGB to Greyscale conversion

__global__ void RGB2GREY(uchar4 *image_RGB, unsigned char *image_GREY, int cols, int rows){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if( x >= cols || y >= rows) {return;} // If dimensions do not match up

  uchar4 rgb = image_RGB[x + y * cols ];
  unsigned char lum =  (0.30f * rgb.x) + (0.59f * rgb.y) + (0.11f * rgb.z); // luminosity effect
  image_GREY[x + y * cols] = lum;

}
