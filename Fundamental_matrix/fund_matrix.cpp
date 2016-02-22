#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h> 
#include <sstream>
#include <iomanip>
#include <stdlib.h> /* exit, EXIT_FAILURE, srand, rand */
#include <time.h>   /* time */

using namespace cv;
using namespace std;

// num of point-correspondence
#define N 8

// Split a single raw stereo image into left and right images
void split_raw_img()
{
	Mat img = imread("Uncalib_stereo.jpg", IMREAD_COLOR);
	int w = 235, h = 220;
	Mat img_left = img(Rect(0, 0, w, h)); //(x, y, w, h)
	Mat img_right = img(Rect(241, 0, w, h)); //(x, y, w, h)
	imwrite("img_left.jpg", img_left);
	imwrite("img_right.jpg", img_right);
}

// Helper function to create a diag matrix from column or row matrix
// diag_elements must be (1xN) or (Nx1)
// returns NxN matrix, type: 32F
Mat myDiag(Mat diag_elements)
{
	int NN;
	if (diag_elements.rows == 1)
	{
		NN = diag_elements.cols;
		Mat ans(NN, NN, CV_32F);
		for (int i = 0; i < NN; i++)
			ans.at<float>(i, i) = diag_elements.at<float>(0, i);
		return ans;
	}
	else if (diag_elements.cols == 1)
	{
		NN = diag_elements.rows;
		Mat ans = Mat::zeros(NN, NN, CV_32F);
		for (int i = 0; i < NN; i++)
			ans.at<float>(i, i) = diag_elements.at<float>(i,0);
		return ans;
	}
	else
	{
		cerr << "Error: argument to myDiag() is neither row- or column- matrix." << endl;
		exit(EXIT_FAILURE);
	}


}

void MyRedCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 2, Scalar(0, 0, 255), thickness, lineType);
}

void MyGreenCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, 2, Scalar(0, 255, 0), thickness, lineType);
}

void MyLine(Mat img, Point start, Point end)
{
	int thickness = 1;
	int lineType = 8;
	line(img, start, end, Scalar(0, 255, 0), thickness, lineType);
}

// Usage: fund_matrix.exe img1 img2 vector_of_corresponding_points
int main(int argc, char** argv)
{
	// Debug version, hard-code all input parameters
	if (argc != 1)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}
	
	Mat img1_color, img2_color;
	img1_color = imread("img_left.jpg", IMREAD_COLOR);
	img2_color = imread("img_right.jpg", IMREAD_COLOR);
	int x_min = 0, x_max = img2_color.cols - 1;
	int y_min = 0, y_max = img2_color.rows - 1;
	Mat img1, img2;
	cvtColor(img1_color, img1, CV_BGR2GRAY);
	cvtColor(img2_color, img2, CV_BGR2GRAY);

	/*
	Point correspondences:
	left    <-> right
	74,39   <-> 82,40 (top left)
	168,35  <-> 176,28 (top right)
	76,175  <-> 63,171 (bottom left)
	197,166 <-> 185,167 (bottom right)
	75,107  <-> 70,106 (left center)
	189,121 <-> 182,119 (right center)
	140,84  <-> 135,81 (upper center)
	122,126 <-> 110,124 (lower center)

	Even indices belong to img1, odd indices belong to img2
	*/
	vector<Point2i> pts(2 * N);
	pts[0] = Point2i(74, 39); pts[1] = Point2i(82, 40);
	pts[2] = Point2i(168, 35); pts[3] = Point2i(176, 28);
	pts[4] = Point2i(76, 175); pts[5] = Point2i(63, 171);
	pts[6] = Point2i(197, 166); pts[7] = Point2i(185, 167);
	pts[8] = Point2i(75, 107); pts[9] = Point2i(70, 106);
	pts[10] = Point2i(189, 121); pts[11] = Point2i(182, 119);
	pts[12] = Point2i(140, 84); pts[13] = Point2i(135, 81);
	pts[14] = Point2i(122, 126); pts[15] = Point2i(110, 124);

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
	//cout << "dx1=" << dx1 << ", dy1=" << dy1 << ", dx2=" << dx2 << ", dy2=" << dy2 << endl;
	//cout << "centroid1=" << centroid1 << endl;
	//cout << "centroid2=" << centroid2 << endl;

	// Normalize to between [-1,1]
	// x' = (x - c.x)/dx
	// y' = (y - c.y)/dy
	// Therefore T = [1/dx,  0,     -c.x/dx; 
	//                 0,    1/dy,  -c.y/dy; 
	//                 0,    0,        1]

	vector<Point2f> pts_norm(2 * N); // Normalized pts
	for (int i = 0; i <= 2 * N - 2; i = i + 2) {
		pts_norm[i] = Point2d((pts[i].x - centroid1.x) / dx1, (pts[i].y - centroid1.y) / dy1);
		pts_norm[i+1] = Point2d((pts[i+1].x - centroid2.x) / dx2, (pts[i+1].y - centroid2.y) / dy2);
	}

	// Construct U matrix
	Mat U(N, 9, CV_32F);
	for (int i = 0; i < N; i++)
	{
		U.at<float>(i, 0) = pts_norm[2 * i + 1].x*pts_norm[2 * i].x;
		U.at<float>(i, 1) = pts_norm[2 * i + 1].x*pts_norm[2 * i].y;
		U.at<float>(i, 2) = pts_norm[2 * i + 1].x;
		U.at<float>(i, 3) = pts_norm[2 * i + 1].y*pts_norm[2 * i].x;
		U.at<float>(i, 4) = pts_norm[2 * i + 1].y*pts_norm[2 * i].y;
		U.at<float>(i, 5) = pts_norm[2 * i + 1].y;
		U.at<float>(i, 6) = pts_norm[2 * i].x;
		U.at<float>(i, 7) = pts_norm[2 * i].y;
		U.at<float>(i, 8) = 1;
	}
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
	
	// SVD decomposition: A = U * diag(w) * VT
	// F, u, vt = [3 x 3], w = [3 x 1]
	Mat w, u, vt;
	SVD::compute(F, w, u, vt, SVD::FULL_UV);
	w.at<float>(w.rows-1,0) = 0; // force last element of w to zero
	Mat F_tilda = u*myDiag(w)*vt;
	Mat T = (Mat_<float>(3,3) << 1/dx1, 0, -centroid1.x/dx1, 0, 1/dy1, -centroid1.y/dy1, 0, 0, 1); // T for img1
	Mat T_prime = (Mat_<float>(3, 3) << 1 / dx2, 0, -centroid2.x / dx2, 0, 1 / dy2, -centroid2.y / dy2, 0, 0, 1); // T for img2
	Mat F_final = T_prime.t() * F_tilda * T;
	//cout << "F_final=" << F_final << endl;

	// Drawing epipolar lines in right image corresponding to keypoints in left image
	// Rule: p_right DOT (F_final * p_left) = 0 => epi_line: F_final * p_left
	// Line equation in right img: px*epi_line[0] + py*epi_line[1] = 0

	for (int i = 0; i < 2*N; i=i+2)
	{
		Vec3f p_left(pts[i].x, pts[i].y, 1);
		Mat epi_line = F_final * Mat(p_left);
		float a = epi_line.at<float>(0, 0); //epi_line[0]
		float b = epi_line.at<float>(1, 0); //epi_line[1]
		float c = epi_line.at<float>(2, 0); //epi_line[2]
		//cout << "a=" << a << ", b=" << b << ", c=" << c << endl;

		MyRedCircle(img1_color, pts[i]);
		MyLine(img2_color, Point2i(x_min, int(-c / b - x_min*a / b)), Point2i(x_max, int(-c / b - x_max*a / b)));
	}

	
	/*
	MyRedCircle(img1_color, pts[0]);
	MyRedCircle(img1_color, pts[2]);
	MyRedCircle(img1_color, pts[4]);
	MyRedCircle(img1_color, pts[6]);
	MyRedCircle(img1_color, pts[8]);
	MyRedCircle(img1_color, pts[10]);
	MyRedCircle(img1_color, pts[12]);
	MyRedCircle(img1_color, pts[14]);
	MyGreenCircle(img2_color, pts[1]);
	MyGreenCircle(img2_color, pts[3]);
	MyGreenCircle(img2_color, pts[5]);
	MyGreenCircle(img2_color, pts[7]);
	MyGreenCircle(img2_color, pts[9]);
	MyGreenCircle(img2_color, pts[11]);
	MyGreenCircle(img2_color, pts[13]);
	MyGreenCircle(img2_color, pts[15]);
	*/

	imshow("left", img1_color);
	moveWindow("left", 0, 0);
	imshow("right", img2_color);
	moveWindow("right", x_max, 0);
	waitKey(0);

	return 0;
}

// To display a float img, first need to convert to char type
// Mat img_out;
// img_float.convertTo(img_out, CV_8U);