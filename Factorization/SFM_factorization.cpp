#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h> 
#include <iomanip>
#include <stdlib.h> /* exit, EXIT_FAILURE, srand, rand */
#include <time.h>   /* time */
#include <utility>  // std::pair, std::make_pair
#include "myCVLib.h"

using namespace cv;
using namespace std;

#define PI 3.14159265
#define R_MIN 25 // radius, not diameter
#define R_MAX 50
#define N 9
#define CENTER_GAP 10
#define R_GAP 10

vector<Point2i> dirs = { Point2i(-1, -1), Point2i(-1, 0), Point2i(-1, 1), Point2i(0, -1), Point2i(0, 0), Point2i(0, 1), Point2i(1, -1), Point2i(1, 0), Point2i(1, 1) };

/*
For complete algorithm, refer to 
Tomasi and Kanade, Shape and motion from image streams under orthography: A factorization method. IJCV, 9(2):137-154, November 1992.
and
Greg Hager's slide Lect10-Multiview.pdf, Page 46
*/

int main(int argc, char** argv)
{
	vector<Point2d> p1,p2;
	readPointsFromFile("set1/pt_2D_1.txt",p1);
	readPointsFromFile("set1/pt_2D_2.txt",p2);
	vector<Point3d> p;
	readPointsFromFile("set1/pt_3D.txt", p);
	if (p1.size() != p2.size())
	{
		cerr << "Error: number of points do not match in the two input files!" << endl;
		return 1;
	}
	int n = (int)p1.size(); // number of point correspondences

	// Compute centroid for frame 1 and 2
	Point2d c1, c2;
	for (int i = 0; i < n; i++)
		c1 += p1[i];
	for (int i = 0; i < n; i++)
		c2 += p2[i];
	c1 /= n;
	c2 /= n;

	// Construct measurement matrix W = (U~, V~), dimension = 2F_rows x P_cols = 4 x 37
	Mat W(4, n, CV_64F); // two frames (F=2)
	for (int i = 0; i < n; i++)
	{
		// U component (x)
		W.at<double>(0, i) = p1[i].x - c1.x;
		W.at<double>(1, i) = p2[i].x - c2.x;
		// V component (y)
		W.at<double>(2, i) = p1[i].y - c1.y;
		W.at<double>(3, i) = p2[i].y - c2.y;
	}

	Mat O1_t, S, O2_t;
	SVD::compute(W.t(), S, O2_t, O1_t, SVD::FULL_UV);
	Mat O1 = O1_t.t();
	Mat O2 = O2_t.t();

	Mat O1_p = O1.colRange(0, 3);
	Mat S_p = myDiag(S.rowRange(0, 3));
	Mat O2_p = O2.rowRange(0, 3);
	
	Mat S_p_sqrt;
	sqrt(S_p, S_p_sqrt);
	Mat R_cap = O1_p * S_p_sqrt; // camera orientation
	Mat S_cap = S_p_sqrt*O2_p; // 3D point position
	
	/*
	cout << "R_cap=\n";
	for (int i = 0; i < R_cap.rows; i++)
		cout << R_cap.rowRange(i, i + 1) << endl;
	cout << "S_cap=\n";
	for (int i = 0; i < S_cap.cols; i++)
		cout << S_cap.colRange(i,i+1).t() << endl;
	*/
	
	/* 
	   Since S_cap is subject to affine ambiguity from tru S, we want to find if our ans is correct.
	   Find the affine transform [A | t] which converts S_cap to true S (from 'set1/pt_3D.txt')
	   Construct A matrix which is 3*37 x 9 -> A has 9-dof, 9 unknowns
	   [X    [ a1 a2 a3     [x       [t1
	    Y  =   a4 a5 a6  *   y    +   t2
	    Z]     a7 a8 a9]     z]       t3]
	   
	   can also be expressed as

	   [X    [ x y z 0 0 0 0 0 0 
	    Y  =   0 0 0 x y z 0 0 0   * [a1 a2 .. a9]' + t
	    Z]     0 0 0 0 0 0 x y z]

		In the code, we call the big matrix M, and solve for 9 elements of A, left hand side = y
		y = M*A + t -> we want to minimize |M*A-y|
	*/
	Mat M(3 * n, 9, CV_64F, Scalar(0)); // initialize to all zero
	for (int i = 0; i < M.rows; i=i+3)
	{
		//double x = S_cap.at<double>(0, i / 3) - S_cap.at<double>(0, 31);
		//double y = S_cap.at<double>(1, i / 3) - S_cap.at<double>(1, 31);
		//double z = S_cap.at<double>(2, i / 3) - S_cap.at<double>(2, 31);
		double x = S_cap.at<double>(0, i / 3);
		double y = S_cap.at<double>(1, i / 3);
		double z = S_cap.at<double>(2, i / 3);
		M.at<double>(i, 0) = x;
		M.at<double>(i, 1) = y;
		M.at<double>(i, 2) = z;
		M.at<double>(i + 1, 3) = x;
		M.at<double>(i + 1, 4) = y;
		M.at<double>(i + 1, 5) = z;
		M.at<double>(i + 2, 6) = x;
		M.at<double>(i + 2, 7) = y;
		M.at<double>(i + 2, 8) = z;
	}
	// Construct y from tru 3D points
	Mat y(3 * n, 1, CV_64F);
	for (int i = 0; i < n; i++)
	{
		y.at<double>(3 * i, 0) = p[i].x;
		y.at<double>(3 * i + 1, 0) = p[i].y;
		y.at<double>(3 * i + 2, 0) = p[i].z;
	}
	Mat A = solve_linear(M, y); // returns 9x1, minimize |M*A-y|

	// Now try to determine the best translation vector t
	Mat conv_pts = M * A;
	conv_pts = conv_pts.reshape(0, n);
	Point3d t(0, 0, 0); // translation vector
	for (int i = 0; i < n; i++)
	{
		Point3d pp(conv_pts.at<double>(i, 0), conv_pts.at<double>(i, 1), conv_pts.at<double>(i, 2));
		t += (p[i] - pp);
	}
	t /= n;

	// Display results and compute mean error
	double err = 0;
	cout << fixed;
	cout.precision(2);
	cout << "         My 3D points            Tru 3D points            err(distance)" << endl;
	for (int i = 0; i < n; i++)
	{
		Point3d pp(conv_pts.at<double>(i, 0), conv_pts.at<double>(i, 1), conv_pts.at<double>(i, 2));
		// align the output for easy viewing
		stringstream convert;
		convert.precision(2);
		convert << fixed << pp + t;
		cout << setw(25) << convert.str();
		stringstream convert2;
		convert2.precision(2);
		convert2 << fixed << p[i];
		cout << setw(25) << convert2.str() <<  "    ->     " << dist(p[i], pp + t) << endl;
		err += dist(p[i], pp+t);
	}
	cout << "-----------" << endl;
	cout << "Average err = " << err / n << endl;

	return 0;
}
