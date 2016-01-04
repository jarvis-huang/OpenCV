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

// num of point-correspondence
#define N 8

// Usage: fund_matrix.exe img1 img2 vector_of_corresponding_points
int main(int argc, char** argv)
{
	// Debug version, hard-code all input parameters
	if (argc != 1)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}
	
	Mat img1, img2;
	img1 = imread("img1", IMREAD_GRAYSCALE);
	img2 = imread("img1", IMREAD_GRAYSCALE);
	
	// Even indices belong to img1, odd indices belong to img2
	vector<Point2i> pts(2*N);
	pts[0] = Point2i(0, 0);
	pts[1] = Point2i(0, 0);
	pts[2] = Point2i(0, 0);
	pts[3] = Point2i(0, 0);
	pts[4] = Point2i(0, 0);
	pts[5] = Point2i(0, 0);
	pts[6] = Point2i(0, 0);
	pts[7] = Point2i(0, 0);
	pts[8] = Point2i(0, 0);
	pts[9] = Point2i(0, 0);
	pts[10] = Point2i(0, 0);
	pts[11] = Point2i(0, 0);
	pts[12] = Point2i(0, 0);
	pts[13] = Point2i(0, 0);
	pts[14] = Point2i(0, 0);
	pts[15] = Point2i(0, 0);

	// Find centroids of samples
	Point2d centroid1(0,0), centroid2(0,0);
	double max_x1 = pts[0].x;
	double max_y1 = pts[0].y;
	double min_x1 = pts[0].x;
	double min_y1 = pts[0].y;
	double max_x2 = pts[1].x;
	double max_y2 = pts[1].y;
	double min_x2 = pts[1].x;
	double min_y2 = pts[1].y;
	for (int i = 0; i <= 2 * N - 2; i = i + 2) {
		centroid1 += (Point2d)pts[i];
		centroid2 += (Point2d)pts[i + 1];
		max_x1 = fmax(max_x1, pts[i].x);
		max_y1 = fmax(max_y1, pts[i].y);
		min_x1 = fmin(min_x1, pts[i].x);
		min_y1 = fmin(min_y1, pts[i].y);
		max_x2 = fmax(max_x2, pts[i+1].x);
		max_y2 = fmax(max_y2, pts[i+1].y);
		min_x2 = fmin(min_x2, pts[i + 1].x);
		min_y2 = fmin(min_y2, pts[i + 1].y);
	}
	centroid1 /= N;
	centroid2 /= N;
	double dx1 = (max_x1 - min_x1) / 2.0;
	double dy1 = (max_y1 - min_y1) / 2.0;
	double dx2 = (max_x2 - min_x2) / 2.0;
	double dy2 = (max_y2 - min_y2) / 2.0;

	// x' = (x - c.x)/dx
	// y' = (y - c.y)/dy
	// Therefore T = [1/dx,  0,     -cx/dx; 
	//                 0,    1/dy,  -cy/dy; 
	//                 0,    0,        1]

	vector<Point2i> pts_norm(2 * N); // Normalized pts
	for (int i = 0; i <= 2 * N - 2; i = i + 2) {
		pts_norm[i] = Point2d((pts[i].x - centroid1.x) / dx1, (pts[i].y - centroid1.y) / dy1);
		pts_norm[i+1] = Point2d((pts[i+1].x - centroid2.x) / dx2, (pts[i+1].y - centroid2.y) / dy2);
	}

	// Construct U matrix
	Mat U(N, 9, CV_32F);
	Mat UTU = U.t() * U;

	// Find eigenvector corresponding to smallest eigenvalue. They are stored in descending order.
	// src*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()
	vector<float> eig_val;
	Mat eig_vec;
	eigen(UTU, eig_val, eig_vec);
	Mat min_eig_vec = eig_vec.row(eig_vec.rows - 1).clone();
	Mat F = min_eig_vec.reshape(0, 3); // 3x3

	// Normalize with L1 norm (max of absolute column sum)
	Mat temp = F.t()*Mat::ones(3, 1, CV_32F);
	float l1norm = fmax(fmax(temp.at<float>(0), temp.at<float>(1)), temp.at<float>(2));
	F = F / l1norm;
	
	// SVD decomposition
	Mat w, u, vt; // w is a 1xN vector, u and vt are square matrices
	SVD::compute(F, w, u, vt, SVD::FULL_UV);
	// cout << "F_size=" << F.size() << "w_size=" << w.size() << "u_size=" << u.size() << "vt_size=" << vt.size();
	w.at<float>(1, w.cols - 1) = 0; // force last element of w to zero
	Mat F_tilda = u*(Mat::eye(F.size(), CV_32F)*w.t())*vt;



	/*
	Mat img_out;
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img);
	waitKey(0);

	corners.clear();
	*/

	return 0;
}

// To display a float img, first need to convert to char type
// Mat img_out;
// img_float.convertTo(img_out, CV_8U);