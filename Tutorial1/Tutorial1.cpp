#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


//////////////////////
// RGB to Gray
//////////////////////
int grb2gray(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: Demo.exe ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat I, img_gray;
	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!I.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cvtColor(I, img_gray, CV_BGR2GRAY);
	//imwrite("fruits_gray.jpg", img_gray);

	namedWindow("Convert2Gray", WINDOW_AUTOSIZE);
	imshow("Convert2Gray", img_gray);
	waitKey(0);
	return 0;
}


//////////////////////
// Linear Blending of two images
//////////////////////
int linearBlend(int argc, char** argv)
{
	if (argc != 3)
	{
		cout << "Usage: Demo.exe image1 image2 (try einstein.tif mri.tif)";
		return -1;
	}
	Mat I1, I2, I3;
	I1 = imread(argv[1], 0);
	I2 = imread(argv[2], 0);
	if (!I1.data || !I2.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	I3 = Mat(I1.size(), I1.type());
	
	namedWindow("Linear Blend", 1);
	for (int i = 100; i >= 0; i--)
	{
		addWeighted(I1, i/100.0, I2, 1-i/100.9, 0.0, I3);
		imshow("Linear Blend", I3);
		waitKey(100);
	}
	waitKey(0);
	return 0;
}

//////////////////////
// Color Channel Extraction
//////////////////////
int colorExtract(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: Demo.exe ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat I, red, green, blue;
	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!I.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	red = Mat(I.size(), CV_8UC1);
	green = Mat(I.size(), CV_8UC1);
	blue = Mat(I.size(), CV_8UC1);

	// Method 1:
	const int channels = I.channels();
	MatIterator_<Vec3b> it, end;
	uchar* p_red = red.data;
	uchar* p_green = green.data;
	uchar* p_blue = blue.data;
	for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
	{
		*p_blue++ = (*it)[0];
		*p_green++ = (*it)[1];
		*p_red++ = (*it)[2];
	}

	// Method 2:
	//Mat out[] = { blue, green, red};
	//int from_to[] = { 0, 0, 1, 1, 2, 2};
	//mixChannels(&I, 1, out, 3, from_to, 3);

	namedWindow("Blue", WINDOW_AUTOSIZE);
	imshow("Blue", blue);
	namedWindow("Green", WINDOW_AUTOSIZE);
	imshow("Green", green);
	namedWindow("Red", WINDOW_AUTOSIZE);
	imshow("Red", red);
	waitKey(0);
	return 0;
}


//////////////////////
// Region of Interest
//////////////////////
int roi(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: Demo.exe ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat I, img_gray;
	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!I.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cvtColor(I, img_gray, CV_BGR2GRAY);

	Mat roi(img_gray, Rect(100, 100, 10, 10));
	cout << "Size of image " << img_gray.size() << endl;
	cout << "Roi = " << roi << endl;
	return 0;
}


//////////////////////
// Bluring
//////////////////////
int blur(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: Demo.exe ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat I, img_gray, img_blured;
	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!I.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cvtColor(I, img_gray, CV_BGR2GRAY);
	blur(img_gray, img_blured, Size(3, 3));
	namedWindow("Blured", WINDOW_AUTOSIZE);
	imshow("Blured", img_blured);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img_gray);
	waitKey(0);
	return 0;
}


//////////////////////
// equalizeHist
//////////////////////
int eqHist(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: Demo.exe ImageToLoadAndDisplay" << endl;
		return -1;
	}
	Mat I, img_gray, img_eqhist;
	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cvtColor(I, img_gray, CV_BGR2GRAY);
	equalizeHist(img_gray, img_eqhist);
	namedWindow("equalizeHist", WINDOW_AUTOSIZE);
	imshow("equalizeHist", img_eqhist);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img_gray);
	waitKey(0);
	return 0;
}


int main(int argc, char** argv)
{

	//double t = (double)getTickCount();
	//// do something ...
	//t = ((double)getTickCount() - t)/getTickFrequency();
	//cout << "Times passed in seconds: " << t << endl;

	return blur(argc, argv);
}