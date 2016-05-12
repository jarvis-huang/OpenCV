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

int main(int argc, char** argv)
{
	//double t = (double)getTickCount();

	if (argc != 2)
	{
		cout << "ERROR: incorrect number or arguments." << endl;
		return -1;
	}

	Mat img = imread(argv[1], IMREAD_COLOR);
	vector<Mat> img_pyr(3);

	for (int i = 0; i < img_pyr.size(); i++)
	{
		if (i == 0)
			img_pyr[i] = img;
		else
			pyrDown(img_pyr[i - 1], img_pyr[i]);

		string window_name = "pyr level " + to_string(i);
		namedWindow(window_name, CV_WINDOW_AUTOSIZE);
		imshow(window_name, img_pyr[i]);
	}

	Mat img_pyr_fullscale;
	resize(img_pyr[1], img_pyr_fullscale, img_pyr[1].size() * 2);
	namedWindow("pyr level 1 at fullscale", CV_WINDOW_AUTOSIZE);
	imshow("pyr level 1 at fullscale", img_pyr_fullscale);
	waitKey(0);

	//cout << "Runtime in seconds: " << ((double)getTickCount() - t) / getTickFrequency() << endl;
	return 0;
}
