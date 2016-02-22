#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define N 8

/*
	Forsyth Book Eq 1.24
*/
int main(int argc, char** argv)
{
	// world points
	int Pw[N][3] = { 
	  { 2, 2, 2 }, 
	  { -2, 2, 2 }, 
	  { -2, 2, -2 }, 
	  { 2, 2, -2 }, 
	  { 2, -2, 2 }, 
	  { -2, -2, 2 }, 
	  { -2, -2, -2 }, 
	  { 2, -2, -2 } 
	};
	// image points
	int Pi[N][2] = {
		{422, 323},
		{178, 323},
		{118, 483},
		{482, 483},
		{438, 73},
		{162, 73},
		{78, 117},
		{522, 117}
	};
	Mat P = Mat::zeros(2 * N, 12, CV_32F);
	for (int r = 0; r < N; r++) {
		P.at<float>(2*r, 0) = Pw[r][0];
		P.at<float>(2*r, 1) = Pw[r][1];
		P.at<float>(2*r, 2) = Pw[r][2];
		P.at<float>(2 * r, 3) = 1;
		P.at<float>(2*r, 4) = 0;
		P.at<float>(2*r, 5) = 0;
		P.at<float>(2*r, 6) = 0;
		P.at<float>(2 * r, 7) = 0;
		P.at<float>(2*r, 8) = -Pw[r][0] * Pi[r][0];
		P.at<float>(2*r, 9) = -Pw[r][1] * Pi[r][0];
		P.at<float>(2*r, 10) = -Pw[r][2] * Pi[r][0];
		P.at<float>(2 * r, 11) = -Pi[r][0];

		P.at<float>(2 * r + 1, 0) = 0;
		P.at<float>(2 * r + 1, 1) = 0;
		P.at<float>(2 * r + 1, 2) = 0;
		P.at<float>(2 * r + 1, 3) = 0;
		P.at<float>(2 * r + 1, 4) = Pw[r][0];
		P.at<float>(2 * r + 1, 5) = Pw[r][1];
		P.at<float>(2 * r + 1, 6) = Pw[r][2];
		P.at<float>(2 * r + 1, 7) = 1;
		P.at<float>(2 * r + 1, 8) = -Pw[r][0] * Pi[r][1];
		P.at<float>(2 * r + 1, 9) = -Pw[r][1] * Pi[r][1];
		P.at<float>(2 * r + 1, 10) = -Pw[r][2] * Pi[r][1];
		P.at<float>(2 * r + 1, 11) = -Pi[r][1];
	}
	cout << "P = " << endl << " " << P << endl << endl;
	Mat rhs = Mat::zeros(2 * N, 12, CV_32F);
	cout << "rhs = " << endl << " " << rhs << endl << endl;
	//Mat M;
	//solve(P, rhs, M, DECOMP_SVD);
	Mat eigenval;
	Mat eigenvec;
	eigen(P.t()*P, eigenval, eigenvec);
	Mat M = eigenvec.row(eigenvec.rows - 1);
	cout << "M = " << endl << " " << M << endl << endl;
	Mat reproj = P*M.t();
	cout << "reproj = " << endl << " " << reproj << endl << endl;
}