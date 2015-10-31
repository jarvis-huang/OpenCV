#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat I, img_gray, img_blured, img_eqhist;
	Mat red, blue, green;

	I = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cvtColor(I, img_gray, CV_BGR2GRAY);
	//imwrite("fruits_gray.jpg", img_gray);

	if (!I.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("Original image", WINDOW_AUTOSIZE);
	imshow("Original image", I);

	double t = (double)getTickCount();
	// do something ...
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Times passed in seconds: " << t << endl;

	//////////////////////
	// RGB to Gray
	//////////////////////
	//namedWindow("Convert2Gray", WINDOW_AUTOSIZE);
	//imshow("Convert2Gray", img_gray);

	//////////////////////
	// Bluring
	//////////////////////
	//blur(img_gray, img_blured, Size(3, 3));
	//namedWindow("Blured", WINDOW_AUTOSIZE);
	//imshow("Blured", img_blured);

	//////////////////////
	// equalizeHist
	//////////////////////
	//equalizeHist(img_gray, img_eqhist);
	//namedWindow("equalizeHist", WINDOW_AUTOSIZE);
	//imshow("equalizeHist", img_eqhist);

	//////////////////////
	// Color Channel Extraction
	//////////////////////
	//red = Mat(I.size(), CV_8UC1);
	//green = Mat(I.size(), CV_8UC1);
	//blue = Mat(I.size(), CV_8UC1);
	// Method 1:
	//const int channels = I.channels();
	//MatIterator_<Vec3b> it, end;
	//uchar* p_red = red.data;
	//uchar* p_green = green.data;
	//uchar* p_blue = blue.data;
	//for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
	//{
	//	*p_blue++ = (*it)[0];
	//	*p_green++ = (*it)[1];
	//	*p_red++ = (*it)[2];
	//}
	// Method 2:
	//Mat out[] = { blue, green, red};
	//int from_to[] = { 0, 0, 1, 1, 2, 2};
	//mixChannels(&I, 1, out, 3, from_to, 3);
	//namedWindow("Blue", WINDOW_AUTOSIZE);
	//imshow("Blue", blue);
	//namedWindow("Green", WINDOW_AUTOSIZE);
	//imshow("Green", green);
	//namedWindow("Red", WINDOW_AUTOSIZE);
	//imshow("Red", red);
	

	//////////////////////
	// Region of Interest
	//////////////////////
	//Mat roi(img_gray, Rect(100, 100, 10, 10));
	//cout << "Size of image " << img_gray.size() << endl;
	//cout << "Roi = " << roi << endl;

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}