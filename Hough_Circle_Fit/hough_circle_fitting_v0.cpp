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

#define PI 3.14159265
#define R_MIN 25 // radius, not diameter
#define R_MAX 50
#define N 9
#define CENTER_GAP 10
#define R_GAP 10

vector<Point2i> dirs = { Point2i(-1, -1), Point2i(-1, 0), Point2i(-1, 1), Point2i(0, -1), Point2i(0, 0), Point2i(0, 1), Point2i(1, -1), Point2i(1, 0), Point2i(1, 1) };

void MyRedCircle(Mat img, Point center, int r=2)
{
	int thickness = -1;
	int lineType = 8;

	circle(img, center, r, Scalar(0, 0, 255), thickness, lineType);
}

void MyGreenCircle(Mat img, Point center, int r=2, int thickness=-1)
{
	//int thickness = -1;
	int lineType = 8;

	circle(img, center, r, Scalar(0, 255, 0), thickness, lineType);
}

void MyLine(Mat img, Point start, Point end)
{
	int thickness = 1;
	int lineType = 8;
	line(img, start, end, Scalar(0, 255, 0), thickness, lineType);
}

// check if p is very close to any existing point in the vector centers, use CENTER_GAP as criteria
bool isDuplicateCenter(vector<Point3i> centers, Point3i p)
{
	if (centers.size() == 0) return false;
	for (auto &c : centers)
	{
		if (abs(c.x - p.x) < CENTER_GAP && abs(c.y - p.y) < CENTER_GAP && abs(c.z - p.z) < R_GAP) return true;
	}
	return false;
}

int main(int argc, char** argv)
{
	// Time the computation
	double t = (double)getTickCount();

	// Debug version, hard-code all input parameters
	if (argc != 1)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}
	
	Mat img_color = imread("coins_small.jpg", IMREAD_COLOR); //coins.jpg
	Mat img, edges;
	cvtColor(img_color, img, CV_BGR2GRAY);
	GaussianBlur(img, img, Size(3, 3),3,3); // remove some edge points inside coins (otherwise there are too many internal edge points should could form a circle, e.g. Franklin's head)
	cv::Canny(img, edges, 180, 240, 3); //100,200,3


	// 3D accumulator array (x * y * r) 
	vector<vector<vector<int>>> A(edges.cols, vector<vector<int>>(edges.rows, vector<int>(R_MAX + 2, 0)));
	for (int x = 0; x < edges.cols; x++)
	{
		for (int y = 0; y < edges.rows; y++)
		{
			if (edges.at<uchar>(y, x) == 255)
			{
				for (int r = R_MIN; r <= R_MAX; r++)
				{
					int prev_xx = 0;
					int prev_yy = 0;
					for (int theta = 0; theta < 360; theta++)
					{
						int xx = x - r*cos(theta*PI / 180);
						int yy = y - r*sin(theta*PI / 180);
						if (xx == prev_xx && yy == prev_yy) continue; // must be 1 pixel distant from prev point
						if (xx < 0 || xx >= edges.cols || yy < 0 || yy >= edges.rows) continue; // must be within bound
						A[xx][yy][r]++;
						prev_xx = xx;
						prev_yy = yy;
					}
				}
			}
		}
	}

	cout << "Accumulator building in seconds: " << ((double)getTickCount() - t) / getTickFrequency() << endl;

	/* Temporarily displaying the accumulator matrix A for debugging
	Mat temp(edges.size(), CV_8U, Scalar(0));
	for (int xx = 0; xx < temp.cols; xx++)
		for (int yy = 0; yy < temp.rows; yy++)
			temp.at<uchar>(yy, xx) = A[xx][yy][29]*2;
	imshow("test", temp);
	imshow("edges", edges);
	waitKey(0);
	return 0;
	*/

	// find local maxima
	vector<Point3i> centers;
	while (centers.size()<N)
	{
		int max_cnt = 0;
		Point3i p(0,0,0);
		for (int x = 0; x < edges.cols; x++)
		{
			for (int y = 0; y < edges.rows; y++)
			{
				for (int r = R_MIN; r <= R_MAX; r++)
				{
					int cnt = 0;
					for (auto &dir : dirs)
					{
						int new_x = x + dir.x;
						int new_y = y + dir.y;
						if (new_x >= 0 && new_x < edges.cols && new_y >= 0 && new_y < edges.rows)
						{
							cnt += A[new_x][new_y][r];
							cnt += A[new_x][new_y][r - 1];
							cnt += A[new_x][new_y][r + 1];
						}
					}
					if (cnt > max_cnt)
					{
						max_cnt = cnt;
						p.x = x; p.y = y; p.z = r;
					}
				}
			}
		}
		if (!isDuplicateCenter(centers, p))
		{
			centers.push_back(p);
			cout << p << endl;	
		}
		for (auto &dir : dirs)
		{
			int new_x = p.x + dir.x;
			int new_y = p.y + dir.y;
			if (new_x >= 0 && new_x < edges.cols && new_y >= 0 && new_y < edges.rows)
			{
				A[new_x][new_y][p.z] = 0;
				A[new_x][new_y][p.z - 1] = 0;
				A[new_x][new_y][p.z + 1] = 0;
			}
		}
	}


	imshow("original", img_color);
	moveWindow("original", 0, 0);
	imshow("edges", edges);
	moveWindow("edges", img.cols, 0);
	// Mark circle centers on image
	for (int k = 0; k < N; k++)
	{
		MyRedCircle(img_color, Point2i(centers[k].x, centers[k].y)); // circle centers (red)
		MyGreenCircle(img_color, Point2i(centers[k].x, centers[k].y), centers[k].z, 1); // circle (green)
	}
	imshow("final", img_color);
	moveWindow("final", 2*img.cols, 0);

	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Times passed in seconds: " << t << endl;

	waitKey(0);

	return 0;
}
