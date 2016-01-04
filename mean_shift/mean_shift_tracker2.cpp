#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <math.h> 

using namespace cv;
using namespace std;

// # of bin per color, a complete feature histo has 3*NBIN bins
// Don't set too small, so that it can accomodate small color variations from frame to frame.
#define NBIN 8
#define BIN_SIZE (256/NBIN)
#define RAD 50 // search radius in pixels
#define H 20 // kernel size
#define CONVERGE_T 0.5 // converge threshold

Point2i ref_p1(1000, 650);
Point2i ref_p2(1140, 870);
int ref_w = ref_p2.x - ref_p1.x;
int ref_h = ref_p2.y - ref_p1.y;

double* h_ref;

double* extractFeature(Mat img, Point2i p1, Point2i p2)
{
	// Ref histo is an array of size 256*3, in the order of B-array, G-array and R-array.
	double* histo = new double[NBIN*3];

	// Initialize to all zeros
	for (int i = 0; i < NBIN * 3; i++)
		*(histo + i) = 0;

	int count = 0;
	for (int x = p1.x; x <= p2.x; x++)
	{
		for (int y = p1.y; y <= p2.y; y++)
		{
			int b = img.at<Vec3b>(y, x)[0] / BIN_SIZE; // blue
			int g = img.at<Vec3b>(y, x)[1] / BIN_SIZE; // green
			int r = img.at<Vec3b>(y, x)[2] / BIN_SIZE; // red
			*(histo + b) += 1;
			*(histo + NBIN + g) += 1;
			*(histo + 2*NBIN + r) += 1;
			count += 3;
		}
	}

	// Normalize so that the whole color histo (across 3 colors) sum up to 1
	for (int i = 0; i < NBIN * 3; i++)
	{
		*(histo + i) /= count;
		//cout << *(histo + i) << " " << endl;
	}

	return histo;
}

double calc_weight(Mat img, Point2i p, double* h_ref, double* h_cur)
{
	Vec3b color = img.at<Vec3b>(p);
	int b = color[0] / BIN_SIZE;
	int g = color[1] / BIN_SIZE;
	int r = color[2] / BIN_SIZE;

	double w = 0;
	w += sqrt(*(h_ref + b) / *(h_cur + b));
	w += sqrt(*(h_ref + NBIN + g) / *(h_cur + NBIN + g));
	w += sqrt(*(h_ref + 2*NBIN + r) / *(h_cur + 2*NBIN + r));

	return w;
}

// ps = starting point, center of the rectangle
Point2i mean_shift_track(Mat img, Point2i ps, double* h_ref)
{
	Point2d p = (Point2d)ps; // center
	cout << "Starting p: " << ps << endl;
	while (true)
	{
		// Get histo of p -> h_cur
		Point2i p1(ceil(p.x - ref_w / 2), ceil(p.y - ref_h / 2));
		Point2i p2(floor(p.x + ref_w / 2), floor(p.y + ref_h / 2));
		double* h_cur = extractFeature(img, p1, p2);

		Point2d meanSum(0,0); // double type since calculation can yield non-integer values
		double kernelSum = 0;

		//   For each point pi in radius=RAD around p
		for (int x = fmax(ceil(ps.x-RAD),0); x <= fmin(floor(ps.x+RAD),img.cols-1); x++)
		{
			for (int y = fmax(ceil(ps.y-RAD),0); y <= fmin(floor(ps.y+RAD),img.rows-1); y++)
			{
				Point2i pi(x, y);
				double d2 = (x - ps.x)*(x - ps.x) + (y - ps.y)*(y - ps.y);

				// Ignore points more than RAD away from center
				if (d2 > RAD*RAD)
					continue;

				// Get weight
				double wi = calc_weight(img, pi, h_ref, h_cur);
				
				cout << "Point: " << pi << ", weight: " << wi << endl;

				// meanSum += wi * pi * g(pi-p)
				meanSum += wi * (Point2d)pi * exp(-0.5*d2 / (H*H));
				// kernelSum += wi * g(pi - p)
				kernelSum += wi * exp(-0.5*d2 / (H*H));
			}
		}
		
		Point2d new_p = meanSum / kernelSum;
		cout << "New p: " << new_p << endl;
		
		// Repeat till converge (new p - old p <= thresh)
		double d_moved = (new_p.x - p.x)*(new_p.x - p.x) + (new_p.y - p.y)*(new_p.y - p.y);
		if (d_moved < CONVERGE_T)
			break;
		else
			p = new_p;
		cout << "Current p: " << p << endl;
	}
	cout << "Final p: " << p << endl;
	return p;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: mean_shift.exe VideoToLoad" << endl;
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

	for (;;)
	{
		vidIn >> frame;
		if (frame.empty())
		{
			cout << " < < <  End of Video!  > > > ";
			break;
		}
		// If first frame, built reference color histogram
		if (count==0)
			h_ref = extractFeature(frame, ref_p1, ref_p2);
		else
		{
			Point2i tracked_c = mean_shift_track(frame, (ref_p1 + ref_p2)/2, h_ref);
			break; // DEBUG ONLY
		}

		//waitKey(0);
		count++;
	}

	return 0;
}