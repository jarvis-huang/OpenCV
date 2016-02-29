#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

#ifndef MYCVLIB 
#define MYCVLIB

void MyRedCircle(Mat img, Point center, int r);

// thickness = -1 means filled circle
void MyGreenCircle(Mat img, Point center, int r, int thickness);

void MyLine(Mat img, Point start, Point end);

// Helper function to create a diag matrix from column or row matrix
// diag_elements must be (1xN) or (Nx1)
// returns NxN matrix, type: CV_64F
Mat myDiag(Mat diag_elements);



// return TRUE if successful, FALSE if unsuccessful
// This version is for Point2d
bool readPointsFromFile(string fname, vector<Point2d>& p);

// Overloaded version for Point3d
bool readPointsFromFile(string fname, vector<Point3d>& p);


// Point distance
double dist(Point2d p1, Point2d p2);
double dist(Point3d p1, Point3d p2);


int count_diff_pixels(cv::Mat a, cv::Mat b);

bool matEq(cv::Mat a, cv::Mat b);

// Linear System Solver
// Problem: A*x = y
// Solve: x
// If y is omitted, it is assumed that y==0 (homogeneous system)
Mat solve_linear(Mat A, Mat y);

#endif