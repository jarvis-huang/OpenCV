#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h> 
#include <sstream>
#include <vector>
#include <iomanip>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979323846
#define N_SCALE 3
#define SIGMA 1.6

// Display the content of a Mat for debugging purposes
void printMat(Mat mat, int prec)
{
	cout << mat.size() << " = [" << endl;
	for (int i = 0; i<mat.size().height; i++)
	{
		for (int j = 0; j<mat.size().width; j++)
		{
			cout << setprecision(prec) << mat.at<float>(i, j);
			if (j != mat.size().width - 1)
				cout << ", ";
			else
				cout << ";" << endl;
		}
	}
	cout << "]" << endl << endl;
}

// kernel width = height = 2*radius+1
// Kernel size must be a param because we will calculate difference of two Gaussian kernels of difference sigma.
// The subtraction requires two kernels to have equal dimension.
Mat my_gaussian(double sigma, int radius)
{
	Mat G(2 * radius + 1, 2 * radius + 1, CV_32F);
	for (int x = -radius; x <= radius; x++)
	{
		for (int y = -radius; y <= radius; y++)
		{
			G.at<float>(x + radius, y + radius) = (float) (1/(2*M_PI*pow(sigma,2.0)) * exp((pow(x,2.0) + pow(y,2.0)) / (-2*pow(sigma,2.0))));
		}
	}
	return G;
}

// kernel width = height = 2*radius+1
// All elements are zero, except the center is 1
// Use to achieve identity (does not change input) during convolution
Mat self_kernel(int radius)
{
	Mat k = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32F);
	k.at<float>(radius, radius) = 1;
	return k;
}


// Default:
//  n_oct = number of octaves
//  n_scale = 3, means how many intermediate scales between si and 2*si
Mat* build_DoG(int n_oct, int n_scale = N_SCALE, float sigma = SIGMA)
{
	/*
		Treat DoG as 2D array of Mat: DoG[n_oct][n_scale+1].
		1st index denotes which octave. 2nd index denotes which scale within that octave.
		However, new int[sizeY][sizeX] requires sizeX to be constant.
		So we implement with 1D array of size n_oct*(n_scale+1).
		DoG[i][j] => Dog[i*(n_scale+1)+j]
		Dog[i][j] = G(2^(i+(j+1)/4)*sigma) - G(2^(i+j/4)*sigma), assume n_scale=3
		For octave[i], lowest scale = 2^i*sigma, highest scale = 2^(i+1)*sigma. There are 3 intermediate scales in btwn.
		Other than octave[0] which has lowest scale = sigma, other octaves have their lowest scale obtained from downsampling the highest scale of previous octave
		therefore no smoothing is performed and kernel should be 1 (matrix with a single 1 in the center)
	*/
	Mat* DoG = new Mat[n_oct*(n_scale+1)]; 
	int idx = 0;
	for (int i = 0; i < n_oct; i++) {
		for (int j = 0; j <= n_scale; j++) {
			double scale_cur = pow(2.0, i+j/(n_scale + 1.0))*sigma;
			double scale_nxt = pow(2.0, i+(j+1)/(n_scale + 1.0))*sigma;
			int r = (int)ceil(scale_nxt*3); // choose size from higher scale Mat, use 3-sigma rule
			Mat G_cur, G_nxt;
			if (i == 0 && j == 0) // first octave first scale
				G_cur = my_gaussian(sigma, r);
			else if (i>0 && j==0)// other octave first scale, form a matrix with a single 1 in the center
				G_cur = self_kernel(r);
			else // any octave non-first scale
				G_cur = my_gaussian(scale_cur, r);
			
			G_nxt = my_gaussian(scale_nxt, r);
			DoG[idx++] = G_nxt - G_cur;
		}
	}
	return DoG;
}

void downsample(const Mat& src, Mat& dst)
{
	dst = Mat(src.rows/2, src.cols/2, src.type());
	for (int r = 0; r < dst.rows; r++) {
		for (int c = 0; c < dst.cols; c++) {
			dst.at<float>(r, c) = src.at<float>(2 * r + 1, 2 * c + 1);
		}
	}
}

void MyFilledCircle(Mat img, Point center)
{ 
	circle(img, center, 1, Scalar(0, 0, 255), 1, 8);
}

// Usage: sift.exe image_file
int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << "ERROR: incorrect number or arguments." << std::endl;
		return -1;
	}
	
	Mat img, img_float;
	img = imread(argv[1], IMREAD_GRAYSCALE);
	img.convertTo(img_float, CV_32F);
	
	//int n_oct = (int)ceil(log2((double)fmin(img.rows, img.cols)));
	int n_oct = 7;
	cout << "building DoG ..." << endl;
	Mat* DoG = build_DoG(n_oct);

	//printMat(DoG[0]*10000, 2);
	//printMat(DoG[1]*10000, 2);
	//printMat(DoG[2]*10000, 2);

	// Convolve DoG with image
	Mat* img_DoG = new Mat[n_oct*(N_SCALE + 1)];
	int idx = 0;
	Mat start_img = img_float;
	Mat temp;
	const Mat GAUSSIAN_TWO_SIGMA = my_gaussian(2 * SIGMA, (int)ceil(2 * SIGMA * 3));

	cout << "convolving with DoG ..." << endl;
	for (int i = 0; i < n_oct; i++) {
		for (int j = 0; j <= N_SCALE; j++) {
			// Another option is to use GaussianBlur() library function
			filter2D(start_img, temp, CV_32F, DoG[idx]);
			img_DoG[idx++] = temp.clone();
		}
		filter2D(start_img, temp, CV_32F, GAUSSIAN_TWO_SIGMA); // apply 2*sigma kernel on lowest scale
		downsample(temp, start_img); // downsample for next octave
	}

	//printMat(img_float(Rect(100, 100, 8, 8)), 3);
	//printMat(img_DoG[0](Rect(100, 100, 8, 8)), 3);
	//printMat(img_DoG[1](Rect(100, 100, 8, 8)), 3);


	// Search for extrema within img_DoG[n_oct][n_scale+1]
	// For each point in img_DoG[i][j], compare with neighboring points within same image, as well as img_DoG[i][j-1] and img_DoG[i][j+1]
	// Never compare images of different octaves.
	// Also lowest and highest scale images within each octave do not produce any output since they don't have lower/higher scale neighbor.
	// The output is keypoint locations (x,y). We use Vector<Point> to store them.
	vector<Point> kp;
	cout << "searching for extrema ..." << endl;
	for (int i = 0; i < n_oct; i++) {
		for (int j = 1; j <= N_SCALE-1; j++) {
			Mat prv = img_DoG[i*(N_SCALE + 1) + j-1];
			Mat cur = img_DoG[i*(N_SCALE + 1) + j];
			Mat nxt = img_DoG[i*(N_SCALE + 1) + j+1];
			Mat prev_cur_nxt[] = { cur, prv, nxt };
			//cout << i << "," << j << "=" << cur.at<float>(7,7) << endl;
			// Skip boundary pixels
			for (int c = 1; c < cur.cols-1; c++) {
				for (int r = 1; r < cur.rows-1; r++) {
					float center = cur.at<float>(r, c);
					bool is_max = false;
					bool is_min = false;
					bool pass = true;

					if (center > prv.at<float>(r, c))
						is_max = true;
					else if (center < prv.at<float>(r, c))
						is_min = true;
					else
						continue;

					for (int x = 0; x <= 2; x++)
					{
						Mat temp = prev_cur_nxt[x];

						float N = temp.at<float>(r - 1, c);
						pass = ((is_min && center < N) || (is_max && center > N));
						if (!pass) break;

						float NE = temp.at<float>(r - 1, c + 1);
						pass = ((is_min && center < NE) || (is_max && center > NE));
						if (!pass) break;

						float E = temp.at<float>(r, c + 1);
						pass = ((is_min && center < E) || (is_max && center > E));
						if (!pass) break;

						float SE = temp.at<float>(r + 1, c + 1);
						pass = ((is_min && center < SE) || (is_max && center > SE));
						if (!pass) break;

						float S = temp.at<float>(r + 1, c);
						pass = ((is_min && center < S) || (is_max && center > S));
						if (!pass) break;

						float SW = temp.at<float>(r + 1, c - 1);
						pass = ((is_min && center < SW) || (is_max && center > SW));
						if (!pass) break;

						float W = temp.at<float>(r, c - 1);
						pass = ((is_min && center < W) || (is_max && center > W));
						if (!pass) break;

						float NW = temp.at<float>(r - 1, c - 1);
						pass = ((is_min && center < NW) || (is_max && center > NW));
						if (!pass) break;
					}

					float p_nxt = nxt.at<float>(r, c);
					pass = pass && ((is_min && center < p_nxt) || (is_max && center > p_nxt));

					if (pass)
						kp.push_back(Point(c, r));
				}
			}
		}
	}

	// To display a float img, first need to convert to char type
	//img_float.convertTo(img_out, CV_8U);

	Mat img_out;
	cvtColor(img, img_out, CV_GRAY2BGR);
	namedWindow("Original", WINDOW_AUTOSIZE);
	imshow("Original", img_out);
	cout << "# of keypoints = " << kp.size() << endl;
	for (int i = 0; i < kp.size(); i++)
		circle(img_out, kp.at(i), 1, Scalar(0, 0, 255), -1, 8);
	namedWindow("KP", WINDOW_AUTOSIZE);
	imshow("KP", img_out);
	waitKey(0);
	
	return 0;
}