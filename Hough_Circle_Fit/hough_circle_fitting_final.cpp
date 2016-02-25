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
#include <utility>  // std::pair, std::make_pair

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

// check if p is very close to any existing point in the vector centers, use CENTER_GAP and R_GAP as criteria
bool isDuplicateCenter(vector<Point3i> centers, Point3i p)
{
	if (centers.size() == 0) return false;
	for (auto &c : centers)
	{
		if (abs(c.x - p.x) < CENTER_GAP && abs(c.y - p.y) < CENTER_GAP && abs(c.z - p.z) < R_GAP) return true;
	}
	return false;
}

// used for sorting according to pair's second key, which is the accumulator count
bool my_compare(pair<Point3i, int> a, pair<Point3i, int> b) 
{
	return (a.second<b.second);
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
	
	Mat img_color = imread("coins.jpg", IMREAD_COLOR);
	Mat img, edges;
	cvtColor(img_color, img, CV_BGR2GRAY);
	GaussianBlur(img, img, Size(3, 3),3,3); // remove some edge points inside coins (otherwise there are too many internal edge points should could form a circle, e.g. Franklin's head)
	cv::Canny(img, edges, 180, 240, 3);


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


	// find local maxima
	vector<pair<Point3i, int>> centers; // Point3i = (x,y,r), int = its count in A
	for (int x = 0; x < edges.cols; x++)
	{
		for (int y = 0; y < edges.rows; y++)
		{
			for (int r = R_MIN; r <= R_MAX; r++)
			{
				bool is_max = true;
				int self_cnt = A[x][y][r];
				if (self_cnt == 0) continue;
				for (auto &dir : dirs)
				{
					int new_x = x + dir.x;
					int new_y = y + dir.y;
					if (new_x >= 0 && new_x < edges.cols && new_y >= 0 && new_y < edges.rows)
						is_max &= (self_cnt >= A[new_x][new_y][r]); // center >= neighbor
				}
				if (is_max)
					centers.push_back(make_pair(Point3i(x, y, r), A[x][y][r]));
			}
		}
	}
	A.clear();

	int C_N = centers.size();
	cout << "# of local maxima = " << C_N << endl;
	// sort according to accumulator count (small to large)
	std::sort(centers.begin(), centers.end(), my_compare);

	// select nine centers with largest count, but avoid inserting duplicates
	vector<Point3i> tru_centers;
	int k = 0;
	while (tru_centers.size() < N)
	{
		Point3i p = centers[C_N - 1 - k].first;
		if (!isDuplicateCenter(tru_centers,p))
			tru_centers.push_back(p);
		k++;
	}
	centers.clear();

	imshow("original", img_color);
	moveWindow("original", 0, 0);
	imshow("edges", edges);
	moveWindow("edges", img.cols, 0);
	// Mark circles and centers on image
	for (int k = 0; k < N; k++)
	{
		MyRedCircle(img_color, Point2i(tru_centers[k].x, tru_centers[k].y)); // circle centers (red)
		MyGreenCircle(img_color, Point2i(tru_centers[k].x, tru_centers[k].y), tru_centers[k].z, 1); // circle (green)
	}
	imshow("final", img_color);
	moveWindow("final", 2*img.cols, 0);
	tru_centers.clear();

	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Total times elapsed in seconds: " << t << endl;

	waitKey(0);

	return 0;
}
