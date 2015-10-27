// Edge_Detect.cpp : Defines the entry point for the console application.
#include "stdafx.h"

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
    uchar r,b,g,mean;
    uchar *data, *data_jvs, *data_red, *data_blue;
	
	IplImage *image = 0, *blured = 0, *cedge = 0, *gray = 0, *edge = 0, *jvs = 0, *red = 0, *blue = 0;
	CvMemStorage* storage;
	CvSeq* lines = 0;
	clock_t start, end;
	double elapsed;

	start = clock();
	// code section to be timed -> begin
	
	if( (image = cvLoadImage( image_file, 1)) == 0 )
        return "";	
	
    storage = cvCreateMemStorage(0);

	blured = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);
    jvs = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1); // yellow & white component

	cvSmooth( image, blured, CV_BLUR, 5, 5, 0, 0 ); // tunable mask size

    data = (uchar*)blured->imageData;
    data_jvs = (uchar*)jvs->imageData;

    
    // Need some error checking for hmin,hmax,wmin,wmax to be in range - jarvis
	for (j=0; j<image->width; j++) // upper half do not care, tunable
    {
		for (i=0; i<floor(image->height*hmin_percent); i++)
		{
			data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0; // masking out uninteresting part
        }
		for (i=floor(image->height*hmax_percent); i<image->height; i++)
        {
            //r = data[i*image->widthStep+j*image->nChannels+2];
            //b = data[i*image->widthStep+j*image->nChannels+0];
            //data_red[i*red->widthStep+j*red->nChannels] = r;
            //data_blue[i*blue->widthStep+j*blue->nChannels] = b;
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


	for (i=floor(image->height*hmin_percent); i<floor(image->height*hmax_percent); i++)
    {
        for (j=floor(image->width*wmin_percent); j<floor(image->width*wmax_percent); j++)
        {
            r = data[i*image->widthStep+j*image->nChannels+2];
			g = data[i*image->widthStep+j*image->nChannels+1];
            b = data[i*image->widthStep+j*image->nChannels+0];
			mean = r/3+g/3+b/3;
            //data_red[i*red->widthStep+j*red->nChannels] = r;
            //data_blue[i*blue->widthStep+j*blue->nChannels] = b;
			if ((r-mean)^2+(g-mean)^2+(b-mean)^2>30) // tunable
			{
				data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 255; // tunable, could do nonlinear transform
			}
			else
			{
				data_jvs[i*jvs->widthStep+j*jvs->nChannels] = 0;
			}
        }
    }

	
	// Create the output image
    cedge = cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 3);

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

	cvZero( cedge );
    // copy edge points
    cvCopy( image, cedge, NULL );

	contours = "";
	for( int i = 0; i < lines->total; i++ )
    {
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		if ( line[0].x==line[1].x || abs((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) > 2  ||
			abs ((double)(line[0].y-line[1].y)/(double)(line[0].x-line[1].x)) < 0.15 ) //tunable
			continue;
        cvLine( cedge, line[0], line[1], CV_RGB(255,0,0), 3, 8 ); // tunable
		ostringstream oss1,oss2,oss3,oss4;
		oss1 << line[0].x;
		oss2 << line[0].y;
		oss3 << line[1].x;
		oss4 << line[1].y;
		contours = contours+"[|"+oss1.str()+","+oss2.str()+"|"+oss3.str()+","+oss4.str()+"]";
    }
	
	
/*
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
*/


	if(openImageWindow){
		cvNamedWindow("orig",1);
		cvNamedWindow("blur",1);
		cvNamedWindow("edge", 1);
		cvNamedWindow("enhance", 1);
		cvShowImage("orig",image);
		cvShowImage("blur",blured);
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

	// code section to be timed -> end
	end = clock();
	elapsed = ((double) (end - start)) / ((double)CLOCKS_PER_SEC);
	//printf("%f",elapsed);
	cout << elapsed << endl;

	return contours;	
}

int main( int argc, char** argv )
{
	char* filename = argc == 2 ? argv[1]:(char*)"pics/depth 10 _edge.jpg";
	ofstream output;
	
	string ct = getContours(filename,1,true,true,0.0,1.0,0.0,1.0);
	

	output.open("output.txt");
	output << ct << endl;
	output.close();

	return 0;
}

#ifdef _EiC
main(1,"edge.c");
#endif