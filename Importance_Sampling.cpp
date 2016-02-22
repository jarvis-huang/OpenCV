#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979323846
#define N 2000 // # of samples, the more the more accurate distribution
#define MAGNIFY 10

/*
Read Chapter of the book by Forsyth and Ponce on the condensation or particle filter.
After you finish reading the chapter, you will find an algorithm for representing
probability distribution functions (pdfs) by sample points and weights (Algorithm
11.5).
a.) Write a matlab function to evaluate the weights of the normal (Gaussian) pdf
b.) Create a random set of 50 sampling points in the interval [-10,10].
Calculate the weights at these points for the normal pdf with the following
parameters (0,2), (1,4), (-3,0.1). Plot the weights against the sampling points.
*/


//Color = BGR
void MyRedCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 5, Scalar(0, 0, 255), thickness, lineType);
}

void MyBlueCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 5, Scalar(255, 0, 0), thickness, lineType);
}

void MyGreenCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 5, Scalar(0, 255, 0), thickness, lineType);
}

void MyLine(Mat img, Point start, Point end)
{
	int thickness = 2;
	int lineType = 8;
	line(img,start,end,Scalar(0, 0, 0),thickness,lineType);
}

//https://en.wikipedia.org/wiki/Normal_distribution
double getWeight(double x, double mean, double var)
{
	return exp(-0.5*pow(x - mean, 2.0) / var) / sqrt(2.0 * var * M_PI);
}

int main(int argc, char** argv)
{
	RNG rng(0xFFFFFFFF);
	double d[N];
	double x;
	for (int i = 0; i < N; i++)
	{
		x = rng.uniform(0x0000, 0x10000); // [a, b)
		x = (x / 0xFFFF) * 20 - 10; // [-10,10]
		d[i] = x;
	}

	// (0, 2), (1, 4), (-3, 0.1)
	double w1[N], w2[N], w3[N];
	for (int i = 0; i < N; i++)
	{
		w1[i] = getWeight(d[i], 0, 2)*20;
		w2[i] = getWeight(d[i], 1, 4)*20;
		w3[i] = getWeight(d[i], -3, 0.1)*20;
		//cout << d[i] << "," << w1[i] << "," << w2[i] << "," << w3[i] << endl;
	}

	//plot
	Mat img(600, 800, CV_8UC3, Scalar(255, 255, 255)); // whiteboard for drawing
	int W = img.cols;
	int H = img.rows;
	MyLine(img, Point2i(0, H / 2), Point2i(W - 1, H / 2)); // X-axis
	MyLine(img, Point2i(W / 2, 0), Point2i(W / 2, H - 1)); // Y-axis
	Point2d O(W / 2, H / 2); // origin
	// we scale x from 10 -> W/3, y -> y * MAGNIFY
	for (int i = 0; i < N; i++)
	{
		Point2d p1 = O + Point2d(d[i] / 10 * W / 3, -w1[i] * MAGNIFY);
		MyRedCircle(img, p1);
		Point2d p2 = O + Point2d(d[i] / 10 * W / 3, -w2[i] * MAGNIFY);
		MyGreenCircle(img, p2);
		Point2d p3 = O + Point2d(d[i] / 10 * W / 3, -w3[i] * MAGNIFY);
		MyBlueCircle(img, p3);
	}
	string title = "Red(0, 2), Green(1, 4), Blue(-3, 0.1)";
	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, img);
	waitKey(0);
}