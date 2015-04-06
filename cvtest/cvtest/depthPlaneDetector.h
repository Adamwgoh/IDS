#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <fstream>
#include <math.h>
#include <time.h>

#include <utility>
#include <iostream>

class depthPlaneDetector{
	//public functions
	public : 
		//constructor
		depthPlaneDetector();
		depthPlaneDetector(cv::Mat* input, int w_ksize, int h_ksize);

		double calcStandardDeviation(cv::Mat* input, cv::Size ksize, int start, int starty);
		cv::Size getWindowsize();
		cv::Mat	getReferenceImage();
		float getStandardDeviation();
		void setWindowsize(cv::Size ksize);
		cv::Mat	searchDeviationDx(cv::Mat input);
		cv::Mat displayDepthGraph(cv::Mat data, int startx,int starty);
	
		void setFoundKeyPt(bool found);
		bool isKeyPtFound();
		int getLeastSquare(cv::Mat src, int win_x, cv::Size window);
		cv::Mat drawPolynomial(cv::Mat graph);

	//private functions
	private : 
		void saveDepthFrame(const char* filename, cv::Mat* image, int framecount);

	//public variables
	public :
		int x,y;

	//private variables
	private : 
		bool foundKeyPt;
		float mean;
		float std_dev;
		cv::Size windowsize;
		cv::Mat* image;
};
