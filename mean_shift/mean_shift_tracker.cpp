#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#define _USE_MATH_DEFINES /* for access to pi constant */
#include <math.h> 

using namespace cv;
using namespace std;

// TO RUN: .\mean_shift.exe .\mean_shift\honey.mp4

#define DEBUG_MODE 0
// Feature = hue histogram. Definition https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma  (Fig. 10)
// First we convert RGB => Hue, which is an angle from 0~2*pi, split into NBIN bins.
// Each pixel vote into one bin based on its hue.
#define NBIN 12
#define BIN_SIZE (360/NBIN) // in degrees
#define H 50 // kernel size
#define CONVERGE_T 0.01 // converge threshold

// Blue cap location in frame0
Point2i ref_p1(1014, 652);
Point2i ref_p2(1086, 672);
int box_w = ref_p2.x - ref_p1.x;
int box_h = ref_p2.y - ref_p1.y;

double* h_ref;

void freeFeature(double* feature)
{
	delete[] feature;
}

double* extractFeature(Mat img, Point2i p1, Point2i p2)
{
	// Ref histo is an array of size 256*3, in the order of B-array, G-array and R-array.
	double* histo = new double[NBIN];

	// Initialize to all zeros
	for (int i = 0; i < NBIN; i++)
		*(histo + i) = 0;

	int count = 0;
	for (int x = p1.x; x <= p2.x; x++)
	{
		for (int y = p1.y; y <= p2.y; y++)
		{
			int b = img.at<Vec3b>(y, x)[0]; // blue
			int g = img.at<Vec3b>(y, x)[1]; // green
			int r = img.at<Vec3b>(y, x)[2]; // red
			double alpha = 0.5*(2 * r - g - b);
			double beta = sqrt(3)*0.5*(g - b);
			double hue = (atan2(beta, alpha) / M_PI + 1) * 180 / BIN_SIZE; // atan2 returns (-pi,+pi)
			int bin = floor(hue + 0.5);
			if (bin >= NBIN)
				bin = 0;
			*(histo + bin) += 1;
			count ++;
		}
	}

	// Normalize so that the whole color histo sum up to 1
	for (int i = 0; i < NBIN; i++)
	{
		*(histo + i) /= count;
		//cout << *(histo + i) << endl;
	}

	return histo;
}

double calc_weight(Mat img, Point2i p, double* h_ref, double* h_cur)
{
	Vec3b color = img.at<Vec3b>(p);
	int b = color[0];
	int g = color[1];
	int r = color[2];
	double alpha = 0.5*(2 * r - g - b);
	double beta = sqrt(3)*0.5*(g - b);
	double hue = (atan2(beta, alpha) / M_PI + 1) * 180 / BIN_SIZE; // atan2 returns (-pi,+pi)
	int bin = floor(hue + 0.5); // We define red-bin to be (-15,+15) degree rather than (0, +30). This steps is to round to nearest bin.
	if (bin >= NBIN)
		bin = 0;

	double w = sqrt(*(h_ref + bin) / *(h_cur + bin));
	
	// This should never happen !!
	if (*(h_cur + bin) == 0)
	{
		cout << "DIVIDE by ZERO!" << endl;
		exit(1);
	}
	return w;
}

// ps = starting point, center of the rectangle
Point2i mean_shift_track(Mat img, Point2i ps, int w, int h, double* h_ref)
{
	Point2d p = (Point2d)ps; // center
	if (DEBUG_MODE)
		cout << "Starting p: " << ps << endl;

	while (true)
	{
		// Get histo of p => h_cur
		Point2i p1(ceil(p.x - w / 2), ceil(p.y - h / 2));
		Point2i p2(floor(p.x + w / 2), floor(p.y + h / 2));
		double* h_cur = extractFeature(img, p1, p2);

		Point2d meanSum(0,0); // double type since calculation can yield non-integer values
		double kernelSum = 0;

		//  For each point pi in the box centered around p
		for (int x = fmax(ceil(p.x - w / 2), 0); x <= fmin(floor(p.x + w/2), img.cols - 1); x++)
		{
			for (int y = fmax(ceil(p.y - h / 2), 0); y <= fmin(floor(p.y + h/2), img.rows - 1); y++)
			{
				Point2i pi(x, y);
				double d2 = (x - p.x)*(x - p.x) + (y - p.y)*(y - p.y);

				// Get weight
				double wi = calc_weight(img, pi, h_ref, h_cur);
				
				// meanSum += wi * pi * g(pi-p)
				meanSum += wi * (Point2d)pi * exp(-0.5*d2 / (H*H));

				//cout << "Point: " << pi << ", weight: " << wi << ", meanSum: " << meanSum << endl;

				// kernelSum += wi * g(pi - p)
				kernelSum += wi * exp(-0.5*d2 / (H*H));
			}
		}
		freeFeature(h_cur);
		Point2d new_p = meanSum / kernelSum;

		if (DEBUG_MODE)
		{
			cout << "meanSum: " << meanSum << endl;
			cout << "kernelSum: " << kernelSum << endl;
			cout << "New p: " << new_p << endl;
		}
		
		// Repeat till converge (new p - old p <= thresh)
		double d_moved = (new_p.x - p.x)*(new_p.x - p.x) + (new_p.y - p.y)*(new_p.y - p.y);
		if (d_moved < CONVERGE_T)
			break;
		else
			p = new_p;
	}

	if (DEBUG_MODE)
		cout << "Final p: " << p << endl;
	
	return (Point2i)p;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: mean_shift.exe VideoFile" << endl;
		return -1;
	}
	VideoCapture vidIn(argv[1]);
	if (!vidIn.isOpened())
	{ 
		cout << "Could not open video input: " << argv[1] << endl;
		return -1; 
	}
	Size refS = Size((int)vidIn.get(CAP_PROP_FRAME_WIDTH), (int)vidIn.get(CAP_PROP_FRAME_HEIGHT));
	int frame_count = vidIn.get(CAP_PROP_FRAME_COUNT);

	Mat frame;
	int count = 0;
	Point2i tracked_center;

	for (;;)
	{
		vidIn >> frame;
		if (frame.empty())
		{
			cout << " < < <  End of Video!  > > > ";
			break;
		}

		// If first frame, built reference color histogram
		if (count == 0)
		{
			h_ref = extractFeature(frame, ref_p1, ref_p2);
			tracked_center = (ref_p1 + ref_p2) / 2;
			namedWindow("track", WINDOW_AUTOSIZE);
			moveWindow("track", 10, 10);
		}
		else
		{
			tracked_center = mean_shift_track(frame, tracked_center, box_w, box_h, h_ref);
			Mat frame_out = frame.clone();
			circle(frame_out, tracked_center, 10, Scalar(0, 0, 255), 2, 8);
			Mat frame_disp;
			// Shrink by factor of 2 for display
			resize(frame_out, frame_disp, Size(), 0.5, 0.5, CV_INTER_AREA);
			imshow("track", frame_disp);
			waitKey(50);
			
			/*
			stringstream ss;
			ss << count;
			imwrite("frame_" + ss.str() + ".jpg", frame_out);
			*/
		}

		if (count == 200)
			break;

		count++;
	}
	freeFeature(h_ref);

	cout << "Total # of frames: " << count << endl;
	return 0;
}