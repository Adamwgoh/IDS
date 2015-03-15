// cvtest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <opencv2\stitching\stitcher.hpp>
#include <sstream>
#include <cmath>
#include <math.h>

#include <utility>
#include <iostream>
#define ROW = 1;
#define COL = 2;

cv::Mat getHistogram(cv::Mat data){
	cv::Mat rawdata = data;
	cv::Mat* graph = new cv::Mat(rawdata.rows, rawdata.cols, CV_8UC1);
	float average_val = 0;
	for (int row = 0; row < rawdata.rows; row++){
		for(int col = 0; col < rawdata.cols; col++){
			//printf("data val is %d\n", rawdata.data[row + (col)*(rawdata.rows)]);
			average_val += rawdata.data[row + (col)*(rawdata.rows)];
			//printf("curr row : %d, curr col : %d, current average val : %f\n", row, col, average_val);
		}

		average_val /= rawdata.rows;
		//printf("average_val : %f\n", average_val);
		graph->data[row + ((int) average_val*rawdata.rows)] = 255;
	}

	return *graph;
}

/**
* calculates the normalized cross-correlation between the reference image and the target image.
* A single run of this function only produces one cross-correlation value for a single translation.
* Normalized cross correlation formula used is : 
* Encc = (E[I0(xi) - ^I0][I1(xi+u) - ^I1])/(Sqrt(E[I0(xi) - ^I0]^2[I1(xi) - ^I1]^2))
*
* Excerpt taken from Computer Vision: Algorithms and Applications by Richard Szeliski
**/
float calcNCC(cv::Mat reference, cv::Mat target, int offx, int offy){
	//this assumes both images are of the same size, as the system uses the same sensor capturing time-series images
	//The offset of the target image from the reference image
	int offsetx = offx;
	int offsety = offy;
	//the referencing image
	cv::Mat ref = reference;//I1
	//the target image
	cv::Mat targ = target;//I0
	int ref_val, targ_val;
	float corr_val, norm_corr_val, ref_nval, targ_nval;
	 
	 //calculate mean of each images
	 for(int i = 0; i < (targ.cols*targ.rows)-3; i++){
		 //target dont move, so dont need to add offset to it

		targ_nval += (targ.data[i] + targ.data[i + 1] + targ.data[i + 2])/3;
	 }
	 //^I = sumof I0(x)/N
	 targ_nval /= targ.elemSize();

	
	 for(int j = offsety; j < ref.cols; j++){
		 for(int i = offsetx; i < ref.rows; i++){
			 //ref is the one only calculating the area of overlapping		 
			 ref_nval += ((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2])/3);
		 }
	 }
	 
	 //mean of number of pixels in the patch
	// printf("lol");
	 
	 ref_nval /= (ref.cols-offsety)*(ref.rows-offsetx);
	 //printf("ref_nval = %f\n", ref_nval);
	 for(int j = offsety; j < ref.cols; j++){
		 for(int i = offsetx; i < ref.rows; i++){
			//value is converted to greyscale first before calculating its difference from the mean
			corr_val += (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval)*
				(((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval);
//			if(i != 0 || j != 0){
				//printf("sum %d, ref_nval %f\n",((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3), ref_nval);
//			}
			//standard deviation of both images
			norm_corr_val += (std::pow( (double) (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval),2)*
				(std::pow( (double) (((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval),2)) );
		 }
	 }
	 //printf("corr val : %f, norm_corr_val : %f\n", corr_val, norm_corr_val);
	float Ecc = corr_val / std::sqrt(norm_corr_val);

	//return value should be in the range of [-1,1]
	return Ecc;
}

/**
*	Calculates normalized cross-correlation between two image with the given window coverage in percentage,
*	that is, the value must be between 0 to 100
*	Output is the highest cross-correlation value within the given window.
**/
float Norm_CrossCorr(cv::Mat imgleft, cv::Mat imgright, int wcoverage, int hcoverage){
	std::vector<float> corr_list = *new std::vector<float>();
	
	float corr = 0;
	//calculate corr based on the percentage of coverage
	for(int offsety = 0; offsety < (imgleft.rows*(hcoverage/100)); offsety++){
		for(int offsetx = 0; offsetx < (imgleft.cols*(wcoverage/100)); offsetx++){
			corr_list.push_back(calcNCC(imgleft, imgright, offsetx, offsety));
		}
	}

	for(int i = 0; i < corr_list.size(); i++){
		if(corr < corr_list.at(i)){
			corr = corr_list.at(i);
		}
	}


	return corr;
}

/**
 * Remaps the image to be in the size of power of 2. This is important as Danielson-Lanczos Lemma needs matrices with size in the power of 2
 * If the input image size is not in the power of 2, zeros-padding will be added to the size of the nearest power of 2
 */
cv::Mat reMap(cv::Mat img){
	cv::Mat image = img;
	int rows = img.rows;
	int cols = img.cols;
	int newrows = 0;
	int newcols = 0;
	

	if((rows & (rows -1) ) != 0){///bitwise element addition not equals 0, which is not of ^2
		//pad 0 to the nearest power of 2 size
		newrows = std::pow(2, ceil(log((float) rows)/log(2.0)) );
	}
	
	if((cols & (cols-1)) != 0){
		newcols = std::pow(2, ceil(log((float) cols)/log(2.0)) );

	}
		if(newrows == 0)	newrows = rows;
		if(newcols == 0)	newcols = cols;
	

	//std::cout << "newrows : " << newrows << ", newcols " << newcols << std::endl;
	image = *new cv::Mat(newrows, newcols, CV_8UC3);

	for(int j = 0; j < img.cols; j++){
		for(int i = 0; i < img.rows; i++){
			image.at<cv::Vec3b>(i,j) = img.at<cv::Vec3b>(i,j);
		}
	}

	for(int j = img.cols; j < newcols; j++){
		for(int i = img.rows; i < newrows; i++){
			//printf("padding with col : %d, row : %d\n",i,j);
			image.at<cv::Vec3b>(i,j)[0] = 0;
			image.at<cv::Vec3b>(i,j)[1] = 0;
			image.at<cv::Vec3b>(i,j)[2] = 0;
		}
	}

	for(int j = 0; j < image.cols; j++){
		for(int	i = 0; i < image.rows; i++){

			//printf("(%d, %d) value r : %d, g : %d, b : %d\n",i, j, image.at<cv::Vec3b>(i,j)[0], image.at<cv::Vec3b>(i,j)[1], image.at<cv::Vec3b>(i,j)[2]);
			//std::cout << (int) image.at<cv::Vec3b>(i,j)[0] << (int)image.at<cv::Vec3b>(i,j)[1] << (int) image.at<cv::Vec3b>(i,j)[2] << std::endl;
		} 
	}

	return image;
}

/**
 * receives an image and do bit reversal on all its values for both dimensions.
 * This refers to the excerpt from Numerical Recipes in C where the splitting of the term is.
 * By doing a right bit reversion, the terms are actually broken down into individual terms that can be applied FFT on.
 * 
 **/
std::vector<int>* bitReversal(std::vector<int> *data ){
	
	int j = 1;
	for(int i = 1; i < data->size(); i++){
		if(j > i){
			//swap
			std::swap(data->at(j), data->at(i));
		}
		int m = data->size()/2;
		while(m >= 2 && j > m){
			j -= m;
			m /= 2;
		}
		j += m;
		
	}

	return data;
}

std::vector<int>* FFT(std::vector<int> *data, bool inverse){
	float two_pi = CV_PI*2;
	int mmax = 1;
	int N = data->size();
	double theta;
	double wtemp;
	double wpr;
	double wpi;
	double wr;
	double wi;
	
	//mmax is the number of times the calculation should be at maximum,
	//once the maximum amount of times to calculate goes over the data size, it stops
	while( N > mmax){//nof point transform	
		//euler's angle value (2Pi/N)
		if(inverse)	theta = two_pi/mmax;	else theta = -two_pi/mmax;		
		//next number of splits and calculations
		int istep = mmax << 1;
		wtemp = std::sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = std::sin(theta);
		wr = 1.0;
		wi = 0.0;
		//half of theta


		printf("Outer loop : mmax : %d, istep : %d\n", mmax, istep);
		//internal loop
		for(int m = 0; m < mmax; m+=2){//size of point transform
			printf("	inner loop1 : m : %d, weight = %f\n", m, wr);
			//cv::waitKey(0);
			//get the Es and Os from the data. 
			for(int i = m; i < N-mmax; i+= istep){
				int j = i + mmax;
				printf("		innerloop1 : i : %d, j = %d\n", i,j);
				double tempreal = wr*data->at(j);
								
				data->at(j) = data->at(i) - tempreal;
				data->at(i) += tempreal;
			}
			wr = (wr*wpr) + wr;
		}
		mmax = istep;

	}//end while

	return data;
}


/**
 *	Perform FFT on a matrix
 **/
cv::Mat* FFT(cv::Mat* data, int width, int height, bool inverse){//data, cols(640), rows(480), false
	//get rows and cols out of Mat and put in these arrays

	std::vector<int>* row = new std::vector<int>(width);
	std::vector<int>* col = new std::vector<int>(height);

	//perform fft on each row can stuff result back into the matrix
	for(int j = 0; j < height; j++){//480
		//TODO: Put your data in here first!
		for(int x = 0; x < width; x++){//640
			printf("x : %d, j : %d\n", x,j);
			row->at(x)  = (int) data->at<uchar>(j,x);
		}
		row = bitReversal(row);
		std::vector<int>* temp_row = FFT(row, false);
		for(int x = 0; x < width; x++){
			printf("x : %d, j : %d\n", x,j);
			data->at<uchar>(j,x) = (uchar) row->at(x);
		}
	}

		//perform fft on each col can stuff result back into the matrix
	for(int i = 0; i < width; i++){
		printf("i : %d\n", i);
		for(int y = 0; y < height; y++){
			row->at(y)  = (int) data->at<uchar>(i,y);
		}

		col = bitReversal(col);
		printf("size of col %d\n", sizeof(col));
		std::vector<int>* temp_col = FFT(col, false);
		for(int j = 0; j < height; j++){	
			printf("i : %d, j : %d\n", i,j);
			data->at<uchar>(i,j) = (uchar) temp_col->at(j);
		}
	}

	return data;
} 


/**
 *
 * OpenCV's stitching method. Not used as it does not perform on the test dataset and have difficulty in extracting
 * the offsets
 **/
cv::Mat stitching(std::vector<cv::Mat> imgs){
	std::vector<cv::Mat> images = imgs;
	cv::Mat result;

	cv::Stitcher lilostitch = cv::Stitcher::createDefault();
	cv::Stitcher::Status s = lilostitch.stitch(images, result);
	std::cout << "status : " << s << std::endl;
	cv::imshow("result", result);

	return result;
}

int main(int argc, TCHAR* argv[])
{
	//target image is the one not moving
	cv::Mat target = cv::imread("rawdata\\setthree_with_markers\\3cframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat target3;
	cv::Mat target6;
	cv::Mat target9;
		//cv::GaussianBlur(target, target9, cv::Size(21,21), 0,0);
	//the reference image that is checked for a certain patch
	cv::Mat img2 = cv::imread("rawdata\\setthree_with_markers\\2cframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat* ref = &img2;

	cv::waitKey(30);
	//cv::imshow("target", target3);
	//cv::imshow("target6", target6);
	//cv::imshow("target9", target9);
	cv::waitKey(0);
	//get each row out and reverse them, then perform FFT

	cv::Mat* result = FFT(ref,ref->cols, ref->rows, false);
	if(result->empty()){
		printf("no result \n");
	}else{
		printf(" lol \n");
	}

	cv::waitKey(30);
	cv::imshow("result", *result);
	cv::waitKey(0);

	
}

