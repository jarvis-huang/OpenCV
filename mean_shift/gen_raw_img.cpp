#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// This program generates a 240x240 boolean image of random blacks dots on white background.
// The density of dots is highest at (100,100).
// This image will be later used to carry out mean shift algorithm.
int main(int argc, char** argv)
{
	// Initialize the image to pure white
	Mat img = Mat::ones(240, 240, CV_8UC1)*255;
	RNG rng(0xFFFFFFFF);
	int xc = 100;
	int yc = 100;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			// Distance to density center (100,100) plus some skew to make probability < 50%
			double d = (abs(i - xc) + abs(j - yc))/2+2;

			// With prob = 1 / d, make this pixel black
			if (rng.uniform(0, 1000) * d < 1000)
				img.at<uchar>(i, j) = 0;
		}
	}
	imwrite("dots.jpg", img);
}