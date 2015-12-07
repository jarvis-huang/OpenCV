#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h> 
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

#define HARRIS_THRESH 0.2   // Used in both methods
#define HARRIS_HI_MARK 12.1 // Used in Method 1 only
#define K 0.05              // parameter for corner detection and edge rejection (see Nixon and Forsyth book)
#define SIGMA 3             // for Gaussian smoothing
#define KERNEL_S (SIGMA*3)  // for Gaussian smoothing
#define WINDOW_S 1         // Hessian is computed from (x-WINDOW_S,y-WINDOW_S) to (x+WINDOW_S,y+WINDOW_S)

// Choose implmentation of Harris corner detector
//   0 -- classic method (Nixon book P191, Forsyth book P150)
//   1 -- Lowe's method using 2nd-order der. (SIFT paper P12)
#define CHOOSE_METHOD 0

// Whether to use canny edge points as corner points (corner chosen from edge points only)
#define USE_CANNY 1
#define lowThreshold 20    // for Canny edge detection
#define ratio 3            // for Canny edge detection
#define aper 3

// Using canny edge detector will improve results because corners are only chosen from edge points.
// However, if keeping the HARRIS_THRESH unchanged, having canny actually increase the # of corners points detected.
// This is because while canny lower the # of initial candidates, "max_score" which is later used to normalize corner points is also lower.


// Usage: harris.exe image_file
// Test images: screwdriver.jpg, chopstix.jpg
int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}
	
	Mat img, img_float, G;
	img = imread(argv[1], IMREAD_GRAYSCALE);
	

	// Pre-filtering
	Mat img_G, img_canny;
	GaussianBlur(img, img_G, Size(KERNEL_S, KERNEL_S), SIGMA);
	img_G.convertTo(img_float, CV_32F, 1 / 255.0);

	// Canny edge detection, returns boolean image. 0-background, 1-edge.
	if (USE_CANNY)
		Canny(img_G, img_canny, lowThreshold, lowThreshold*ratio, aper);

	// Compute gradient
	Mat mat_dx = Mat::zeros(img_float.size(), CV_32F);
	Mat mat_dy = Mat::zeros(img_float.size(), CV_32F);
	for (int x = 0; x < img_float.cols; x++) {
		for (int y = 0; y < img_float.rows; y++) {
			if (x==0)
				mat_dx.at<float>(y, x) = img_float.at<float>(y, x + 1) - img_float.at<float>(y, x);
			else if (x == img_float.cols-1)
				mat_dx.at<float>(y, x) = img_float.at<float>(y, x) - img_float.at<float>(y, x - 1);
			else
				mat_dx.at<float>(y, x) = 0.5*(img_float.at<float>(y, x + 1) - img_float.at<float>(y, x - 1));

			if (y==0)
				mat_dy.at<float>(y, x) = img_float.at<float>(y + 1, x) - img_float.at<float>(y, x);
			else if (y == img_float.rows-1)
				mat_dy.at<float>(y, x) = img_float.at<float>(y, x) - img_float.at<float>(y - 1, x);
			else
				mat_dy.at<float>(y, x) = 0.5*(img_float.at<float>(y + 1, x) - img_float.at<float>(y - 1, x));
		}
	}

	// We want to compute Hessian = [Dxx Dxy; Dxy; Dyy] for each pixel
	// Store detected corner points in a vector
	vector<Point2i> corners;
	Mat harris_scores = Mat::zeros(img_float.size(), CV_32F); // for storing Harris scores
	float max_score = 0;
	int edge_cnt = 0;

	for (int x = 1; x < img_float.cols-1; x++) {
		for (int y = 1; y < img_float.rows-1; y++) {
			if (USE_CANNY)
			{
				// Skip if not an edge point
				if (img_canny.at<char>(y, x) == 0)
					continue;
			}

			float Dxx = 0, Dyy = 0, Dxy = 0;
			if (CHOOSE_METHOD==0)
			{
				for (int di = -WINDOW_S; di <= WINDOW_S; di++) {
					for (int dj = -WINDOW_S; dj <= WINDOW_S; dj++) {
						Dxx += mat_dx.at<float>(y + dj, x + di)*mat_dx.at<float>(y + dj, x + di);
						Dyy += mat_dy.at<float>(y + dj, x + di)*mat_dy.at<float>(y + dj, x + di);
						Dxy += mat_dx.at<float>(y + dj, x + di)*mat_dy.at<float>(y + dj, x + di);
					}
				}
			}
			else
			{
				Dxx = img_float.at<float>(y, x + 1) + img_float.at<float>(y, x - 1) - 2 * img_float.at<float>(y, x);
				Dyy = img_float.at<float>(y + 1, x) + img_float.at<float>(y - 1, x) - 2 * img_float.at<float>(y, x);
				Dxy = 0.25*(img_float.at<float>(y + 1, x + 1) + img_float.at<float>(y - 1, x - 1) - img_float.at<float>(y + 1, x - 1) - img_float.at<float>(y - 1, x + 1));
			}

			float det = Dxx*Dyy - Dxy*Dxy;
			float trace = Dxx + Dyy;

			float score1 = det - K / 4.0 * trace*trace;
			float score2 = trace*trace / det;

			// Score must be positive to prevent normalization problems.
			if (CHOOSE_METHOD == 0) {
				if (det > 0 && score1 > 0) {
					harris_scores.at<float>(y, x) = score1;
					max_score = fmax(max_score, score1);
				}
			}
			else
			{
				if (det > 0 && score1 > 0 && score2 < HARRIS_HI_MARK) {
					harris_scores.at<float>(y, x) = score1;
					max_score = fmax(max_score, score1);
				}
			}
			edge_cnt++;
		}
	}

	// Normalize to [0,1]
	for (int x = 1; x < img_float.cols - 1; x++) {
		for (int y = 1; y < img_float.rows - 1; y++) {
			// Magnify by taking sqrt
			harris_scores.at<float>(y, x) = sqrt(harris_scores.at<float>(y, x)/max_score);
		}
	}

	for (int x = 1; x < img_float.cols - 1; x++) {
		for (int y = 1; y < img_float.rows - 1; y++) {
			float score = harris_scores.at<float>(y, x);

			// Score must be local maxima
			float score_N = harris_scores.at<float>(y - 1, x);
			float score_S = harris_scores.at<float>(y + 1, x);
			float score_E = harris_scores.at<float>(y, x + 1);
			float score_W = harris_scores.at<float>(y, x - 1);

			if (score >= fmax(fmax(score_N, score_S), fmax(score_E, score_W)) && score > HARRIS_THRESH)
				corners.push_back(Point(x, y));
		}
	}

	cout << "CHOOSE_METHOD = " << CHOOSE_METHOD << endl;
	cout << "USE_CANNY = " << USE_CANNY << endl;
	if (USE_CANNY)
		cout << "# of edge points detected = " << edge_cnt << endl;
	cout << "# of corner points detected = " << corners.size() << endl;

	Mat img_out;
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);
	harris_scores.convertTo(img_out, CV_8U, 255.0); // scale back to [0,255]
	namedWindow("Corner Map", WINDOW_AUTOSIZE);
	imshow("Corner Map", img_out); // Lighter regions are corners (high curvature).
	cvtColor(img, img, CV_GRAY2BGR);
	for (int i = 0; i < corners.size(); i++)
		circle(img, corners.at(i), 1, Scalar(0, 0, 255), -1, 8);
	namedWindow("Corners", WINDOW_AUTOSIZE);
	imshow("Corners", img);
	if (USE_CANNY)
	{
		namedWindow("Canny", WINDOW_AUTOSIZE);
		imshow("Canny", img_canny);
	}
	waitKey(0);

	corners.clear();
	return 0;
}

// To display a float img, first need to convert to char type
// Mat img_out;
// img_float.convertTo(img_out, CV_8U);