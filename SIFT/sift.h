#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define M_PI 3.14159265358979323846
#define N_OCT 7
#define N_SCALE 3
#define SIGMA 1.6
#define PEAK_THRESH 7.65 //0.03*255=7.65
#define EDGE_THRESH 12.1 //(r+1)^2/r, where r=10

// Display the content of a Mat for debugging purposes
void printMat(Mat mat, int prec)
{
	std::streamsize old_precision = cout.precision();
	cout << mat.size() << " = [" << endl;
	cout << fixed;
	cout.precision(prec);
	for (int i = 0; i<mat.size().height; i++)
	{
		for (int j = 0; j<mat.size().width; j++)
		{
			cout << mat.at<float>(i, j);
			if (j != mat.size().width - 1)
				cout << ", ";
			else
				cout << ";" << endl;
		}
	}
	cout << "]" << endl << endl;
	cout.setf(2, ios::floatfield);
	cout.precision(old_precision);
}

void MyFilledCircle(Mat img, Point center)
{
	circle(img, center, 1, Scalar(0, 0, 255), 1, 8);
}


// calculate the cofactor of element (row,col)
int GetMinor(float **src, float **dest, int row, int col, int order)
{
	// indicate which col and row is being copied to dest
	int colCount = 0, rowCount = 0;

	for (int i = 0; i < order; i++)
	{
		if (i != row)
		{
			colCount = 0;
			for (int j = 0; j < order; j++)
			{
				// when j is not the element
				if (j != col)
				{
					dest[rowCount][colCount] = src[i][j];
					colCount++;
				}
			}
			rowCount++;
		}
	}

	return 1;
}

// Calculate the determinant recursively.
double CalcDeterminant(float **mat, int order)
{
	// order must be >= 0
	// stop the recursion when matrix is a single element
	if (order == 1)
		return mat[0][0];

	// the determinant value
	float det = 0;

	// allocate the cofactor matrix
	float **minor;
	minor = new float*[order - 1];
	for (int i = 0; i<order - 1; i++)
		minor[i] = new float[order - 1];

	for (int i = 0; i < order; i++)
	{
		// get minor of element (0,i)
		GetMinor(mat, minor, 0, i, order);
		// the recusion is here!

		det += (i % 2 == 1 ? -1.0 : 1.0) * mat[0][i] * CalcDeterminant(minor, order - 1);
		//det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
	}

	// release memory
	for (int i = 0; i<order - 1; i++)
		delete[] minor[i];
	delete[] minor;

	return det;
}

// matrix inversioon
// the result is put in Y
void MatrixInversion(float **A, int order, float **Y)
{
	// get the determinant of a
	double det = 1.0 / CalcDeterminant(A, order);

	// memory allocation
	float *temp = new float[(order - 1)*(order - 1)];
	float **minor = new float*[order - 1];
	for (int i = 0; i<order - 1; i++)
		minor[i] = temp + (i*(order - 1));

	for (int j = 0; j<order; j++)
	{
		for (int i = 0; i<order; i++)
		{
			// get the co-factor (matrix) of A(j,i)
			GetMinor(A, minor, j, i, order);
			Y[i][j] = det*CalcDeterminant(minor, order - 1);
			if ((i + j) % 2 == 1)
				Y[i][j] = -Y[i][j];
		}
	}

	// release memory
	//delete [] minor[0];
	delete[] temp;
	delete[] minor;
}

void downsample(const Mat& src, Mat& dst)
{
	dst = Mat(src.rows / 2, src.cols / 2, src.type());
	for (int r = 0; r < dst.rows; r++) {
		for (int c = 0; c < dst.cols; c++) {
			dst.at<float>(r, c) = src.at<float>(2 * r + 1, 2 * c + 1);
		}
	}
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
			G.at<float>(x + radius, y + radius) = (float)(1 / (2 * M_PI*pow(sigma, 2.0)) * exp((pow(x, 2.0) + pow(y, 2.0)) / (-2 * pow(sigma, 2.0))));
		}
	}
	return G;
}

// returns an array of Mat, size = n_oct*(N_SCALE+2)
void build_gaussian_pyramid(const Mat& img, Mat* dst, int n_oct)
{
	int idx = 0;
	Mat start_img = img.clone();
	Mat temp;

	for (int oct = 0; oct < n_oct; oct++) {
		for (int s = 0; s < N_SCALE + 2; s++) {
			// For 2nd octave and above, the lowest scale image doesn't need filtering. It's just downsampled from previous octave.
			if (oct > 0 && s==0) {
				dst[idx++] = start_img.clone();
				continue;
			}
			float sigma = pow(2.0, oct + s / (N_SCALE + 1.0))*SIGMA;
			int r = (int)ceil(sigma * 3);
			Mat G = my_gaussian(sigma, r);
			filter2D(start_img, temp, CV_32F, G);
			dst[idx++] = temp.clone();
		}
		start_img.deallocate();
		downsample(temp, start_img); // lowest scale of octave[n+1] = highest scale of scale[n] downsampled by 2
	}
}

void build_gradient(const Mat* pyramid, Mat* Grad_x, Mat* Grad_y, int n_oct)
{
	Mat L;

	for (int idx = 0; idx < n_oct*(N_SCALE + 2); idx++) {
		L = pyramid[idx];
		Grad_x[idx] = L.clone();
		Grad_y[idx] = L.clone();
		for (int r = 0; r < L.rows; r++) {
			for (int c = 0; c < L.cols; c++) {
				// First column
				if (c == 0) {
					Grad_x[idx].at<float>(r, c) = L.at<float>(r, c + 1) - L.at<float>(r, c);
				} // Last column
				else if (c == L.cols - 1) {
					Grad_x[idx].at<float>(r, c) = L.at<float>(r, c) - L.at<float>(r, c-1);
				} // Middle columns
				else
					Grad_x[idx].at<float>(r, c) = 0.5*(L.at<float>(r, c + 1) - L.at<float>(r, c - 1));
				
				// First row
				if (r == 0) {
					Grad_y[idx].at<float>(r, c) = L.at<float>(r + 1, c) - L.at<float>(r, c);
				}
				else if (r == L.rows - 1) {
					Grad_y[idx].at<float>(r, c) = L.at<float>(r, c) - L.at<float>(r-1, c);
				}
				else {
					Grad_y[idx].at<float>(r, c) = 0.5*(L.at<float>(r + 1, c) - L.at<float>(r - 1, c));
				}
			}
		}
	}
}

void build_mag_orient(const Mat* Grad_x, const Mat* Grad_y, Mat* mag, Mat* theta, int n_oct)
{
	Mat g_x, g_y;
	for (int idx = 0; idx < n_oct*(N_SCALE + 2); idx++) {
		g_x = Grad_x[idx];
		g_y = Grad_y[idx];
		mag[idx] = g_x.clone();
		theta[idx] = g_x.clone();
		for (int r = 0; r < g_x.rows; r++) {
			for (int c = 0; c < g_x.cols; c++) {
				mag[idx].at<float>(r, c) = 2 * sqrt(g_x.at<float>(r, c)*g_x.at<float>(r, c) + g_y.at<float>(r, c)*g_y.at<float>(r, c));
				theta[idx].at<float>(r, c) = atan2(g_y.at<float>(r, c), g_x.at<float>(r, c));
			}
		}
	}
}