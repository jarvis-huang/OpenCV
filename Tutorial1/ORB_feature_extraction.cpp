#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
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


int main(int argc, const char* argv[])
{

	if (argc != 2)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}

	Mat img = imread(argv[1], IMREAD_GRAYSCALE);

	vector<KeyPoint> keypoints;
	Mat desc;

	MyTimer timer(true);

	//Ptr<AKAZE> akaze = AKAZE::create();
	//akaze->detectAndCompute(img, noArray(), keypoints, desc);

	Ptr<ORB> orb = ORB::create();
	orb->setMaxFeatures(100);
	orb->detectAndCompute(img, noArray(), keypoints, desc);

	cout << "Runtime in seconds: " << timer.stop() << endl;

	// Add results to image
	Mat output;
	drawKeypoints(img, keypoints, output);

	namedWindow("ORB output", CV_WINDOW_AUTOSIZE);
	imshow("ORB output", output);
	waitKey(0);

	
	return 0;
}
