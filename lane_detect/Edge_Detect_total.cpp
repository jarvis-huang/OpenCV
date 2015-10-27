// Edge_Detect.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include "stdlib.h"
#include <string>
using std::string;

#include <iostream>
using std::cerr;
using std::endl;
using namespace std;

#include <fstream>
using std::ofstream;

#include <sstream>
using std::ostringstream;


#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <ctime>
#include <Windows.h>

#endif


string getContours(char* image_file,int fileNumber, bool writeToFile, bool openImageWindow, double hmin_percent, double hmax_percent, double wmin_percent, double wmax_percent )
{
    string contours;
	int edge_thresh = 80;
	int i,j;
    uchar r,b,g,rup,bup,gup,rleft,bleft,gleft,mean;
	int dev;
    uchar *data, *data_eq, *data_jvs, *data_red, *data_blue, *data_green;
	
	IplImage *image = 0, *image_eq = 0, *eq = 0, *blured = 0, *cedge = 0, *gray = 0, *edge = 0, *jvs = 0, *red = 0, *blue = 0, *green = 0;
	CvMemStorage* storage;
	CvSeq* lines = 0;
	CvSeq* lines2 = 0;
	
	if( (image = cvLoadImage( image_file, 1)) == 0 )
        return "";	
	
    storage = cvCreateMemStorage(0);
	
	eq = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
	blured = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);
	image_eq = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);
    jvs = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1); // yellow & white component
	red = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
	green = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
	blue = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);

	
	cvSmooth( image, blured, CV_BLUR, 3, 3, 0, 0 ); // tunable mask size

    data = (uchar*)blured->imageData;
	data_eq = (uchar*)image_eq->imageData;
    data_jvs = (uchar*)jvs->imageData;
	data_red = (uchar*)red->imageData;
	data_green = (uchar*)green->imageData;
	data_blue = (uchar*)blue->imageData;
    
    // Need some error checking for hmin,hmax,wmin,wmax to be in range - jarvis
	for (j=0; j<image->width; j++)
	{
		for (i=0; i<image->height; i++)
		{
			r = data[i*blured->widthStep+j*blured->nChannels+2];
			g = data[i*blured->widthStep+j*blured->nChannels+1];
            b = data[i*blured->widthStep+j*blured->nChannels+0];
            data_red[i*red->widthStep+j*red->nChannels] = r;
            data_blue[i*blue->widthStep+j*blue->nChannels] = b;
			data_green[i*green->widthStep+j*green->nChannels] = g;
		}
	}

	cvEqualizeHist(red,eq);
	cvCopy( eq, red, NULL );	
	cvEqualizeHist(green,eq);
	cvCopy( eq, green, NULL );
	cvEqualizeHist(blue,eq);
	cvCopy( eq, blue, NULL );

	for (j=0; j<image->width; j++)
	{
		for (i=0; i<image->height; i++)
		{
			data_eq[i*image->widthStep+j*image->nChannels+2] = data_red[i*red->widthStep+j*red->nChannels];
			data_eq[i*image->widthStep+j*image->nChannels+1] = data_green[i*green->widthStep+j*green->nChannels];;
			data_eq[i*image->widthStep+j*image->nChannels+0] = data_blue[i*blue->widthStep+j*blue->nChannels];;
		}
	}

	for (j=0; j<image->width; j++) // upper half do not care, tunable
    {
		for (i=0; i<floor(image->height*hmin_percent); i++)
		{
			data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0; // masking out uninteresting part
        }
		for (i=floor(image->height*hmax_percent); i<image->height; i++)
        {
			data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0; // masking out uninteresting part
        }
    }
	for (i=floor(image->height*hmin_percent); i<floor(image->height*hmax_percent); i++) // upper half do not care,tunable
    {
        for (j=0; j<floor(image->width*wmin_percent); j++)
        {
			data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0; // masking out uninteresting part
        }
		for (j=floor(image->width*wmax_percent); j<image->width; j++)
        {
			data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0; // masking out uninteresting part
        }
    }
	

	contours = "";
	cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);
	cvZero( cedge );
	cvCopy( image, cedge, NULL );


	// *********** Yellow_line detect ************
	CvSeq* lines_store[4];
	CvSeq* lines2_store[4];
	float dev_rho_store[4], dev_theta_store[4];
	int count_store[4];

	for (int cl=3; cl<=3; cl++)
	{
		for (i=floor(image->height*hmin_percent); i<floor(image->height*hmax_percent); i++)
		{
			for (j=floor(image->width*wmin_percent); j<floor(image->width*wmax_percent); j++)
			{
				if (cl == 3)
				{
					r = data_eq[i*image->widthStep+j*image->nChannels+2];
					g = data_eq[i*image->widthStep+j*image->nChannels+1];
					b = data_eq[i*image->widthStep+j*image->nChannels+0];
					if (r/1.2>b && g/1.2>b && r>50 && b>50)
						data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 255;
					else
						data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0;
				}
				else
				{
					r = data_eq[i*image->widthStep+j*image->nChannels+cl];
					data_jvs[i*jvs->widthStep+j*jvs->nChannels] = r;
				}
			}
		}

		
		// Create the output image
		//cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);

		// Convert to grayscale
		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);

		// Create a window
	    
		//cvSmooth( jvs, edge, CV_BLUR, 3, 3, 0, 0 );
		cvNot( jvs, edge );

		// Run the edge detector on grayscale
		cvCanny(jvs, edge, (float)edge_thresh/2, (float)edge_thresh*2, 3);

		//cvShowImage("edge", edge);
		lines = cvHoughLines2( edge, storage, CV_HOUGH_PROBABILISTIC, 4, CV_PI/36, 80, 20, 20 ); // tunable
		lines2 = cvHoughLines2( edge, storage, CV_HOUGH_STANDARD, 4, CV_PI/36, 80, 0, 0 );
		lines_store[cl] = lines;
		lines2_store[cl] = lines2;

		//cvZero( cedge );
		// copy edge points
		//cvCopy( image, cedge, NULL );

		float sum_rho=0,sum_theta=0,mean_rho,mean_theta,dev_rho=0,dev_theta=0;
		for( int i = 0; i < lines2->total; i++ )
		{
			float* line = (float*)cvGetSeqElem(lines2,i);
			sum_rho += line[0];
			sum_theta += line[1];
		}
		mean_rho = sum_rho/lines2->total;
		mean_theta = sum_theta/lines2->total;
		for( int i = 0; i < lines2->total; i++ )
		{
			float* line = (float*)cvGetSeqElem(lines2,i);
			dev_rho += (line[0]-mean_rho)*(line[0]-mean_rho);
			dev_theta += (line[1]-mean_theta)*(line[1]-mean_theta);
		}

		dev_rho_store[cl] = dev_rho = dev_rho/lines2->total;
		dev_theta_store[cl] = dev_theta = dev_theta/lines2->total;

		cout << dev_rho << ", " << dev_theta << endl;

		for( int i = 0; i < lines->total; i++ )
		{
			CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
			if ( line[0].x==line[1].x || abs((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) > 5  ||
				abs ((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) < 0.15 ) //restrict slope, tunable
				continue;
			// restrict slope based on left/right half
			if (line[0].x/2+line[1].x/2>image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) < 0)
				continue;
			if (line[0].x/2+line[1].x/2<image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) > 0)
				continue; 
			count_store[cl]++;
		}

		if (dev_rho<20000 && dev_theta<0.8) // must have small deviation
		{
			for( int i = 0; i < lines->total; i++ )
			{
				CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
				if ( line[0].x==line[1].x || abs((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) > 5  ||
					abs ((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) < 0.15 ) //restrict slope, tunable
					continue;
				// restrict slope based on left/right half
				if (line[0].x/2+line[1].x/2>image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) < 0)
					continue;
				if (line[0].x/2+line[1].x/2<image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) > 0)
					continue; 
				cvLine( cedge, line[0], line[1], CV_RGB(255,0,0), 3, 8 ); // tunable
				ostringstream oss1,oss2,oss3,oss4;
				oss1 << line[0].x;
				oss2 << line[0].y;
				oss3 << line[1].x;
				oss4 << line[1].y;
				contours = contours+"[|"+oss1.str()+","+oss2.str()+"|"+oss3.str()+","+oss4.str()+"]";
			}
		}
	}

	// ********** Combining three colors ****************
	if (contours.size() < 1000)
	{
		for (int cl=0; cl<=2; cl++)
		{
			for (i=floor(image->height*hmin_percent); i<floor(image->height*hmax_percent); i++)
			{
				for (j=floor(image->width*wmin_percent); j<floor(image->width*wmax_percent); j++)
				{
					r = data[i*image->widthStep+j*image->nChannels+cl];
					data_jvs[i*jvs->widthStep+j*jvs->nChannels] = r;
				}
			}

			
			// Create the output image
			//cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);

			// Convert to grayscale
			gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
			edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
			cvCvtColor(image, gray, CV_BGR2GRAY);

			// Create a window
		    
			//cvSmooth( jvs, edge, CV_BLUR, 3, 3, 0, 0 );
			cvNot( jvs, edge );

			// Run the edge detector on grayscale
			cvCanny(jvs, edge, (float)edge_thresh/2, (float)edge_thresh*2, 3);

			//cvShowImage("edge", edge);
			lines = cvHoughLines2( edge, storage, CV_HOUGH_PROBABILISTIC, 4, CV_PI/36, 150, 20, 20 ); // tunable
			lines2 = cvHoughLines2( edge, storage, CV_HOUGH_STANDARD, 4, CV_PI/36, 150, 0, 0 );

			//cvZero( cedge );
			// copy edge points
			//cvCopy( image, cedge, NULL );

			float sum_rho=0,sum_theta=0,mean_rho,mean_theta,dev_rho=0,dev_theta=0;
			
			for( int i = 0; i < lines2->total; i++ )
			{
				float* line = (float*)cvGetSeqElem(lines2,i);
				sum_rho += line[0];
				sum_theta += line[1];
			}
			mean_rho = sum_rho/lines2->total;
			mean_theta = sum_theta/lines2->total;
			for( int i = 0; i < lines2->total; i++ )
			{
				float* line = (float*)cvGetSeqElem(lines2,i);
				dev_rho += (line[0]-mean_rho)*(line[0]-mean_rho);
				dev_theta += (line[1]-mean_theta)*(line[1]-mean_theta);
			}

			dev_rho_store[cl] = dev_rho = dev_rho/lines2->total;
			dev_theta_store[cl] = dev_theta = dev_theta/lines2->total;
			cout << dev_rho << ", " << dev_theta << endl;
			if (dev_rho<20000 && dev_theta<0.8)
			{
				for( int i = 0; i < lines->total; i++ )
				{
					CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
					if ( line[0].x==line[1].x || abs((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) > 5  ||
						abs ((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) < 0.15 ) //restrict slope, tunable
						continue;
					if (line[0].x/2+line[1].x/2>image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) < 0)
						continue;
					if (line[0].x/2+line[1].x/2<image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) > 0)
						continue; 
					cvLine( cedge, line[0], line[1], CV_RGB(255,0,0), 3, 8 ); // tunable
					ostringstream oss1,oss2,oss3,oss4;
					oss1 << line[0].x;
					oss2 << line[0].y;
					oss3 << line[1].x;
					oss4 << line[1].y;
					contours = contours+"[|"+oss1.str()+","+oss2.str()+"|"+oss3.str()+","+oss4.str()+"]";
				}
			}
		}
	}

	
	// ****************** gray closeness ***************
	if (contours.size() < 1000)
	{
		for (i=floor(image->height*hmin_percent); i<floor(image->height*hmax_percent); i++)
		{
			for (j=floor(image->width*wmin_percent); j<floor(image->width*wmax_percent); j++)
			{
				r = data_eq[i*image->widthStep+j*image->nChannels+2];
				g = data_eq[i*image->widthStep+j*image->nChannels+1];
				b = data_eq[i*image->widthStep+j*image->nChannels+0];
				mean = r/3+g/3+b/3;
				dev = abs(r-mean)+abs(g-mean)+abs(b-mean);
				
				if ( dev > 30) // tunable
				{
					data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 255; // tunable, could do nonlinear transform
				}
				else
				{
					data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0;
				}
			}
		}

		gray = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		edge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
		cvCvtColor(image, gray, CV_BGR2GRAY);

		// Create a window
	    
		//cvSmooth( jvs, edge, CV_BLUR, 3, 3, 0, 0 );
		cvNot( jvs, edge );

		// Run the edge detector on grayscale
		cvCanny(jvs, edge, (float)edge_thresh/2, (float)edge_thresh*2, 3);

		//cvShowImage("edge", edge);
		lines = cvHoughLines2( edge, storage, CV_HOUGH_PROBABILISTIC, 4, CV_PI/36, 150, 20, 20 ); // tunable
		lines2 = cvHoughLines2( edge, storage, CV_HOUGH_STANDARD, 4, CV_PI/36, 150, 0, 0 );

		//cvZero( cedge );
		// copy edge points
		//cvCopy( image, cedge, NULL );

		float sum_rho=0,sum_theta=0,mean_rho,mean_theta,dev_rho=0,dev_theta=0;
		for( int i = 0; i < lines2->total; i++ )
		{
			float* line = (float*)cvGetSeqElem(lines2,i);
			sum_rho += line[0];
			sum_theta += line[1];
		}
		mean_rho = sum_rho/lines2->total;
		mean_theta = sum_theta/lines2->total;
		for( int i = 0; i < lines2->total; i++ )
		{
			float* line = (float*)cvGetSeqElem(lines2,i);
			dev_rho += (line[0]-mean_rho)*(line[0]-mean_rho);
			dev_theta += (line[1]-mean_theta)*(line[1]-mean_theta);
		}

		cout << "->" << dev_rho << ", " << dev_theta << endl;

		if (dev_rho<100000 && dev_theta<10)
		{
			for( int i = 0; i < lines->total; i++ )
			{
				CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
				if ( line[0].x==line[1].x || abs((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) > 5  ||
					abs ((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) < 0.15 ) //restrict slope, tunable
					continue;
				if (line[0].x/2+line[1].x/2>image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) < 0)
					continue;
				if (line[0].x/2+line[1].x/2<image->width/2 && (double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x) > 0)
					continue; 
				cvLine( cedge, line[0], line[1], CV_RGB(255,0,0), 3, 8 ); // tunable
				ostringstream oss1,oss2,oss3,oss4;
				oss1 << line[0].x;
				oss2 << line[0].y;
				oss3 << line[1].x;
				oss4 << line[1].y;
				cout << "Hello" << endl;
				contours = contours+"[|"+oss1.str()+","+oss2.str()+"|"+oss3.str()+","+oss4.str()+"]";
			}
		}
	}


	/*
	if (contours.size() < 200)
	{
		int least,less,more,most,temp;
		
		int most_count = max(max(count_store[0],count_store[1]),max(count_store[2],count_store[3]));
		int least_count = min(max(count_store[0],count_store[1]),min(count_store[2],count_store[3]));
		for (int i=0; i<=3; i++)
		{
			if (count_store[i] == most_count)
				most = i;
			if (count_store[i] == least_count)
				least = i;
		}
		for (int i=0; i<=3; i++)
		{
			if (i != most && i != least)
				less = i;
		}
		for (int i=0; i<=3; i++)
		{
			if (i != most && i != least && i != less)
				more = i;
		}
		if (count_store[less] > count_store[more])
		{
			temp = more;
			more = less;
			less = temp;
		}
		// start
		int index_to_use[4];
	}
	*/





	if(writeToFile)
	{
		char edge_file_name[50];
        sprintf(edge_file_name ,"%s",image_file);
		string edgeFileName = edge_file_name;
		int pos_point = edgeFileName.find(".",0);
		edgeFileName = edgeFileName.erase(pos_point);
		edgeFileName += "_edge.jpg";
		cvSaveImage(edgeFileName.c_str(), cedge);
	

		char ctr_file_name[50];
        sprintf(ctr_file_name ,"%s",image_file);
		string ctrFileName = ctr_file_name;
		pos_point = ctrFileName.find(".",0);
		ctrFileName = ctrFileName.erase(pos_point);
		ctrFileName += ".contours";
		FILE* ctrf =fopen( ctrFileName.c_str(), "w+" );
		char* ctrs = new char[contours.length()]; 
        strcpy(ctrs,contours.c_str());
		fprintf( ctrf, "%s\n", ctrs );
        if(ctrs != NULL) fclose( ctrf );
	}

	

	if(openImageWindow){
		cvNamedWindow("orig",1);
		cvNamedWindow("blur",1);
		cvNamedWindow("histeq",1);
		cvNamedWindow("edge", 1);
		cvNamedWindow("enhance", 1);
		cvShowImage("orig",image);
		cvShowImage("blur",blured);
		cvShowImage("histeq",image_eq);
		cvShowImage("enhance", jvs);
		cvShowImage("edge", cedge);
		cvWaitKey(0);
	}



    // Wait for a key stroke; the same function arranges events processing
   
    cvReleaseImage(&image);
    cvReleaseImage(&gray);
    cvReleaseImage(&edge);
	cvReleaseImage(&cedge);
	cvReleaseImage(&jvs);

	if(openImageWindow){
		cvDestroyWindow("edge");
		cvDestroyWindow("enhance");
	}
	
	return contours;	
}

int main( int argc, char** argv )
{
	
	char * stored_filename[20] =
	{
		"pics/depth_0030.ppm",
		"pics/depth_0031.ppm",
		"pics/depth_0032.ppm",
		"pics/depth_0033.ppm",
		"pics/depth_0034.ppm",
		"pics/depth_0040.ppm",
		"pics/depth_0041.ppm",
		"pics/depth_0042.ppm",
		"pics/depth_0043.ppm",
		"pics/depth_0044.ppm",
		"pics/depth_0045.ppm",
		"pics/depth_0046.ppm",
		"pics/depth_0047.ppm",
		"pics/depth_0048.ppm",
		"pics/depth_0049.ppm",
		"pics/depth_0060.ppm",
		"pics/depth_0061.ppm",
		"pics/depth_0062.ppm",
		"pics/depth_0063.ppm",
		"pics/depth_0064.ppm"
	};

	for (int index = 0; index < 1; index++)
	{
		//char* filename = argc == 2 ? argv[1]:stored_filename[index];
		char* filename = argc == 2 ? argv[1]:"pics/depth_0011.ppm";
		ofstream output;
		clock_t start, end;
		double elapsed;

		start = clock();

		// process to be timed -> begins here
		string ct = getContours(filename,1,true,true,0.0,1.0,0.0,1.0);
		output.open("output.txt");
		output << ct << endl;
		output.close();
		// process to be timed -> ends here
		end = clock();
		elapsed = ((double) (end - start)) / ((double)CLOCKS_PER_SEC);
		//printf("%f",elapsed);
		cout << elapsed << endl;
	}

	return 0;
}

#ifdef _EiC
main(1,"edge.c");
#endif