#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h> 

using namespace cv;
using namespace std;

void mean_shift(Mat I, double r, double thresh)
{
	int h = I.size().height;
	int w = I.size().width;
	RNG rng(0xFFFFFFF9); // Good: 0xFFFFFFF7
	Point2d pos(rng.uniform(0, w + 1), rng.uniform(0, h + 1)); // start at a random location
	bool converged = false;
	Mat img_out; // for displaying output 
	cout << "Curent center location " << pos << endl;
	while (!converged)
	{
		Point2d meanSum(0, 0);
		double kernelSum = 0;
		int xc = pos.x;
		int yc = pos.y;
		for (int x = fmax(xc - r,0); x <= fmin(xc + r,w-1); x++)
		{
			for (int y = fmax(yc - r,0); y <= fmin(yc + r,h-1); y++)
			{
				// At all sample locations within the circle
				if (I.at<uchar>(y, x) == 0)
				{
					meanSum += exp(-0.5*((x - xc)*(x - xc) + (y - yc)*(y - yc))/(r*r)) * Point2d(x, y);
					kernelSum += exp(-0.5*((x - xc)*(x - xc) + (y - yc)*(y - yc)) / (r*r));
				}
			}
		}

		Point2d new_pos = meanSum / kernelSum;
		converged = ((new_pos.x - pos.x)*(new_pos.x - pos.x) + (new_pos.y - pos.y)*(new_pos.y - pos.y) < thresh);
		pos = new_pos;

		cvtColor(I, img_out, CV_GRAY2BGR);
		circle(img_out, pos, r, Scalar(0, 0, 255), 1, 8);
		namedWindow("img", WINDOW_AUTOSIZE);
		imshow("img", img_out);
		waitKey(2000);
		cout << "Curent center location " << pos << endl;
	}
	cout << "Final center location " << pos << endl;
	waitKey(0);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: mean_shift.exe ImageToLoad" << endl;
		return -1;
	}
	Mat I, img_gray, img_eqhist;
	I = imread(argv[1], 0);
	mean_shift(I,35,1); // larger radius is more likely to escape from local maxima
	return 0;
}