#include "myCVLib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h> 

using namespace cv;
using namespace std;

void MyRedCircle(Mat img, Point center, int r = 2)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, r, Scalar(0, 0, 255), thickness, lineType);
}

// thickness = -1 means filled circle
void MyGreenCircle(Mat img, Point center, int r = 2, int thickness = -1)
{
	int lineType = 8;

	circle(img, center, r, Scalar(0, 255, 0), thickness, lineType);
}

void MyLine(Mat img, Point start, Point end)
{
	int thickness = 1;
	int lineType = 8;
	line(img, start, end, Scalar(0, 255, 0), thickness, lineType);
}

// Helper function to create a diag matrix from column or row matrix
// diag_elements must be (1xN) or (Nx1)
// returns NxN matrix, type: CV_64F
Mat myDiag(Mat diag_elements)
{
	int NN;
	if (diag_elements.rows == 1)
	{
		NN = diag_elements.cols;
		Mat ans(NN, NN, CV_64F);
		for (int i = 0; i < NN; i++)
			ans.at<double>(i, i) = diag_elements.at<double>(0, i);
		return ans;
	}
	else if (diag_elements.cols == 1)
	{
		NN = diag_elements.rows;
		Mat ans = Mat::zeros(NN, NN, CV_64F);
		for (int i = 0; i < NN; i++)
			ans.at<double>(i, i) = diag_elements.at<double>(i, 0);
		return ans;
	}
	else
	{
		cerr << "Error: argument to myDiag() is neither row- or column- matrix." << endl;
		exit(EXIT_FAILURE);
	}


}


// return TRUE if successful, FALSE if unsuccessful
// This version is for Point2d
bool readPointsFromFile(string fname, vector<Point2d>& p)
{
	string line;
	ifstream myfile(fname);
	bool first_line = true;
	int n, i;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			if (first_line)
			{
				first_line = false;
				n = std::stoi(line);
				p.resize(n);
				i = 0;
			}
			else
			{
				double x, y;
				x = std::stod(line.substr(0, line.find(" ")));
				y = std::stod(line.substr(line.find(" ") + 1));
				p[i] = Point2d(x, y);
				i++;
			}
		}
		myfile.close();
		return true;
	}
	else
	{
		cout << "Unable to open file";
		return false;
	}
}

// Overloaded version for Point3d
bool readPointsFromFile(string fname, vector<Point3d>& p)
{
	string line;
	ifstream myfile(fname);
	bool first_line = true;
	int n, i;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			if (first_line)
			{
				first_line = false;
				n = std::stoi(line);
				p.resize(n);
				i = 0;
			}
			else
			{
				double x, y, z;
				unsigned first_space = (unsigned)line.find(" ");
				unsigned second_space = (unsigned)line.find_last_of(" ");
				x = std::stod(line.substr(0, first_space));
				y = std::stod(line.substr(first_space + 1, second_space - first_space - 1));
				z = std::stod(line.substr(second_space + 1));
				p[i] = Point3d(x, y, z);
				i++;
			}
		}
		myfile.close();
		return true;
	}
	else
	{
		cout << "Unable to open file";
		return false;
	}
}


double dist(Point2d p1, Point2d p2)
{
	return norm(Vec2d(p1), Vec2d(p2));
}

double dist(Point3d p1, Point3d p2)
{
	return norm(Vec3d(p1), Vec3d(p2));
}

int count_diff_pixels(cv::Mat a, cv::Mat b)
{
	if (a.type() != b.type())
	{
		cerr << "Error: arguments passed to count_diff_pixels() have different types!";
		exit(EXIT_FAILURE);
	}
	if (a.size() != b.size())
	{
		cerr << "Error: arguments passed to count_diff_pixels() have different sizes!";
		exit(EXIT_FAILURE);
	}
	cv::Mat diff;
	cv::compare(a, b, diff, cv::CMP_NE);
	return cv::countNonZero(diff);
}


bool matEq(cv::Mat a, cv::Mat b) 
{
	// False if type of size mismatch
	if (a.type() != b.type() || a.size() != b.size())
		return false;

	cv::Mat diff;
	// location where a!=b will be marked 255 in diff, 0 elsewhere
	// if a==b, diff will be filled with 0, no NonZero elements
	cv::compare(a, b, diff, cv::CMP_NE);
	return cv::countNonZero(diff) == 0;
}

Mat solve_linear(Mat A, Mat y)
{
	if (A.type() != y.type())
	{
		cerr << "Error: arguments passed to solve_linear() have different types!";
		exit(EXIT_FAILURE);
	}
	if (A.rows != y.rows)
	{
		cerr << "Error: in solve_linear(), A and y must have equal number of rows!";
		exit(EXIT_FAILURE);
	}



	Mat allzero = Mat::zeros(y.size(), y.type());
	if (matEq(y,allzero)) // y==0, homogeneous system, use solveZ()
	{
		Mat x;
		SVD::solveZ(A, x);
		return x;
	}
	else // y!=0
	{
		SVD my_SVD(A);
		Mat x;
		my_SVD.backSubst(y, x);
		return x;

		/*
		////Naive method////
		SVD::compute(A, w, u, vt, SVD::FULL_UV);
		u = u.colRange(0, A.cols);
		Mat x = vt.t()*myDiag(w).inv()*u.t()*y;
		*/
	}
}