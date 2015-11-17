#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <math.h> 
#include <sstream>
#include <vector>
#include <iomanip>
#include "sift.h"

using namespace cv;
using namespace std;


// TODO:
//	1. Move memory allocation to inside functions. Use (Mat*&) as function param type.




// kernel width = height = 2*radius+1
// All elements are zero, except the center is 1
// Use to achieve identity (does not change input) during convolution
Mat self_kernel(int radius)
{
	Mat k = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32F);
	k.at<float>(radius, radius) = 1;
	return k;
}






// n = Descriptor array width (n*n array)
// r = # of orientation bins
// m = # of neiboring samples per key point (m*m samples)
void build_orientation_hist(const Mat& img, const vector<Point>& kp, const vector<Point>& kp_scale, vector<float*> descriptor, int n_oct, int n, int r, int m)
{

	for (int i = 0; i < n_oct; i++) {
		for (int j = 0; j <= N_SCALE; j++) {
			Mat G;
			float sigma = pow(2.0, i + j / (N_SCALE + 1.0))*SIGMA;
			int width = ceil(3 * sigma) * 2 + 1;
			GaussianBlur(img, G, cvSize(width, width), sigma);
			// Another option is to use GaussianBlur() library function
			//filter2D(start_img, temp, CV_32F, DoG[idx]);
		}
	}
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
	
	// Option:
	//     int n_oct = (int)ceil(log2((double)fmin(img.rows, img.cols)));
	int n_oct = 7;
	Mat* g_pyramid = new Mat[n_oct*(N_SCALE + 2)];
	cout << "building Gaussian pyramid ..." << endl;
	build_gaussian_pyramid(img_float, g_pyramid, N_OCT);
	cout << "building DoG ..." << endl;
	Mat* DoG = new Mat[n_oct*(N_SCALE + 1)];
	build_DoG(g_pyramid, DoG, n_oct);

	//printMat(DoG[0](Rect(100, 100, 8, 8)), 3);


	// Search for extrema within DoG[n_oct][n_scale+1]
	// For each point in DoG[i][j], compare with neighboring points within same image, as well as DoG[i][j-1] and DoG[i][j+1]
	// Never compare images of different octaves.
	// Also lowest and highest scale images within each octave do not produce any output since they don't have lower/higher scale neighbor.
	// The output is keypoint locations (x,y). We use Vector<Point> to store them.
	vector<Point> kp;
	vector<Point> kp_scale; // stores the scale index (i,j), actual scale = pow(2.0, i+j/(N_SCALE+1.0))*sigma
	cout << "searching for extrema ..." << endl;
	for (int i = 0; i < n_oct; i++) {
		for (int j = 1; j <= N_SCALE-1; j++) {
			Mat prv = DoG[i*(N_SCALE + 1) + j - 1];
			Mat cur = DoG[i*(N_SCALE + 1) + j];
			Mat nxt = DoG[i*(N_SCALE + 1) + j + 1];
			Mat prev_cur_nxt[] = { cur, prv, nxt };

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

					if (pass) {
						kp.push_back(Point(c, r));
						kp_scale.push_back(Point(i, j));
					}
				}
			}
		}
	}

	cout << "# of keypoints (before refinement) = " << kp.size() << endl;

	// Eliminate keypoints along edges
	// Now we have two vectors: kp, kp_scale
	// We want to compute Hessian = [Dxx Dxy; Dxy; Dyy] for each kp
	// img_DoG
	vector<Point> refined_kp;
	vector<Point> refined_kp_scale;
	for (int i = 0; i < kp.size(); i++) {
		Point loc = kp.at(i);
		int x = loc.x;
		int y = loc.y;
		Point scale = kp_scale.at(i);
		Mat img_cur = DoG[scale.x*(N_SCALE + 1) + scale.y];
		float D = img_cur.at<float>(y, x);
		float Dx = 0.5*(img_cur.at<float>(y, x + 1) - img_cur.at<float>(y, x - 1));
		float Dy = 0.5*(img_cur.at<float>(y + 1, x) - img_cur.at<float>(y - 1, x));
		float Dxx = img_cur.at<float>(y, x + 1) + img_cur.at<float>(y, x - 1) - 2 * img_cur.at<float>(y, x);
		float Dyy = img_cur.at<float>(y+1, x) + img_cur.at<float>(y-1, x) - 2 * img_cur.at<float>(y, x);
		float Dxy = 0.25*(img_cur.at<float>(y + 1, x + 1) + img_cur.at<float>(y - 1, x - 1) - img_cur.at<float>(y + 1, x - 1) - img_cur.at<float>(y - 1, x + 1));
		Mat img_higher = DoG[scale.x*(N_SCALE + 1) + scale.y + 1];
		Mat img_lower = DoG[scale.x*(N_SCALE + 1) + scale.y - 1];
		float Ds = 0.5*(img_higher.at<float>(y, x) - img_lower.at<float>(y, x));
		float Dss = img_higher.at<float>(y, x) + img_lower.at<float>(y, x) - 2 * img_cur.at<float>(y, x);
		float Dxs = 0.25*(img_higher.at<float>(y, x + 1) + img_lower.at<float>(y, x - 1) - img_higher.at<float>(y, x - 1) - img_lower.at<float>(y, x + 1));
		float Dys = 0.25*(img_higher.at<float>(y + 1, x) + img_lower.at<float>(y - 1, x) - img_higher.at<float>(y - 1, x) - img_lower.at<float>(y + 1, x));	

		float* lapD[3];
		lapD[0] = new float[3]{Dxx, Dxy, Dxs};
		lapD[1] = new float[3]{Dxy, Dyy, Dys};
		lapD[2] = new float[3]{Dxs, Dys, Dss};

		float* inv_lapD[3];
		for (int x = 0; x < 3; ++x)
			inv_lapD[x] = new float[3];

		MatrixInversion((float**)lapD, 3, (float**)inv_lapD);
		double peak_x = -inv_lapD[0][1] * Dx - inv_lapD[0][1] * Dy - inv_lapD[0][2] * Ds;
		double peak_y = -inv_lapD[1][1] * Dx - inv_lapD[1][1] * Dy - inv_lapD[1][2] * Ds;
		double peak_s = -inv_lapD[2][1] * Dx - inv_lapD[2][1] * Dy - inv_lapD[2][2] * Ds;
		float peak_val = D + 0.5*(Dx*peak_x + Dy*peak_y + Ds*peak_s);
		
		// Eliminate kp if peak_val<PEAK_THRESH (low contrast) or score>EDGE_THRESH (along an edge)
		if (abs(peak_val) < PEAK_THRESH)
			continue;

		float score = (Dxx + Dyy)*(Dxx + Dyy) / (Dxx*Dyy - Dxy*Dxy);
		if (score > EDGE_THRESH)
			continue;

		// Pass, store in new vector
		refined_kp.push_back(Point(loc));
		refined_kp_scale.push_back(Point(scale));
	}
	kp.clear();
	kp_scale.clear();

	cout << "# of keypoints (after refinement) = " << refined_kp.size() << endl;

	/*
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
	*/

	

	Mat* Grad_x = new Mat[n_oct*(N_SCALE + 2)];
	Mat* Grad_y = new Mat[n_oct*(N_SCALE + 2)];
	cout << "building gradient ..." << endl;
	build_gradient(g_pyramid, Grad_x, Grad_y, N_OCT);
	Mat* mag = new Mat[n_oct*(N_SCALE + 2)];
	Mat* theta = new Mat[n_oct*(N_SCALE + 2)];
	cout << "building magnitude and orientation ..." << endl;
	build_mag_orient(Grad_x, Grad_y, mag, theta, N_OCT);
	vector<float> angles(refined_kp.size());
	Point one_kp = refined_kp.at(0);
	Point one_kp_scale = refined_kp_scale.at(0);
	get_kp_orient(mag, theta, one_kp, one_kp_scale, angles, N_OCT);


	waitKey(0);
	return 0;
}

// To display a float img, first need to convert to char type
// Mat img_out;
// img_float.convertTo(img_out, CV_8U);