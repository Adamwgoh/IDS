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
	if(ref.channels() != targ.channels()){
		std::cout << "two images have different number of channels!\n" << std::endl;
		return 0;
	}
	int ref_val, targ_val;
	float corr_val, norm_corr_val, ref_nval, targ_nval;
	
	if(ref.channels() == 1){
		 //calculate mean of each images
		 for(int i = 0; i < (targ.cols*targ.rows)-3; i++){
			//target dont move, so dont need to add offset to it

			targ_nval += (int) targ.data[i];
		 }

		 //^I = sumof I0(x)/N
		 targ_nval /= targ.elemSize();

	
		 for(int j = offsety; j < ref.cols; j++){
			 for(int i = offsetx; i < ref.rows; i++){
				 //ref is the one only calculating the area of overlapping		 
				 ref_nval += (int) ref.at<uchar>(i,j);
			 }
		 }	
		
	}else if(ref.channels() == 3){
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
	}

	 

	 //mean of number of pixels in the patch
	// printf("lol");

	 if(ref.channels() == 1){
	

	 	 ref_nval /= (ref.cols-offsety)*(ref.rows-offsetx);

		 for(int j = offsety; j < ref.cols; j++){
			 for(int i = offsetx; i < ref.rows; i++){
				corr_val += ((int) ref.at<uchar>(i,j) - ref_nval)*((int) targ.at<uchar>(i,j) - targ_nval);
				norm_corr_val += ( std::pow((double) ((int) ref.at<uchar>(i,j)) ,2) ) * ( std::pow((double) ((int) targ.at<uchar>(i-offsetx,j-offsety) - targ_nval),2) );
			 }
		 }
	 }else if(ref.channels() == 3){

		 ref_nval /= (ref.cols-offsety)*(ref.rows-offsetx);
		 //printf("ref_nval = %f\n", ref_nval);
		 for(int j = offsety; j < ref.cols; j++){
			 for(int i = offsetx; i < ref.rows; i++){
				//value is converted to greyscale first before calculating its difference from the mean
				corr_val += (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval)*
					(((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval);

				//standard deviation of both images
				norm_corr_val += (std::pow( (double) (((ref.at<cv::Vec3b>(i,j)[0] + ref.at<cv::Vec3b>(i,j)[1] + ref.at<cv::Vec3b>(i,j)[2]) /3) - ref_nval),2)*
					(std::pow( (double) (((targ.at<cv::Vec3b>(i-offsetx,j-offsety)[0] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[1] + targ.at<cv::Vec3b>(i-offsetx,j-offsety)[2])/3) - targ_nval),2)) );
			 }
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
float Norm_CrossCorr(cv::Mat imgleft, cv::Mat imgright, int startx, int starty, int wcoverage, int hcoverage){
	std::vector<float> corr_list = *new std::vector<float>();
	
	float corr = 0;
	//calculate corr based on the percentage of coverage
	for(int offsety = startx; offsety < (imgleft.rows*(hcoverage/100)); offsety++){
		for(int offsetx = starty; offsetx < (imgleft.cols*(wcoverage/100)); offsetx++){
			float corrval = calcNCC(imgleft, imgright, offsetx, offsety);

			corr_list.push_back(corrval);
		}
	}

	for(int i = 0; i < corr_list.size(); i++){
		if(corr < corr_list.at(i)){
			corr = corr_list.at(i);
		}
	}

	printf(" biggest corr : %f\n", corr);
	return corr;
}

/**
 * Remaps the image to be in the size of power of 2. This is important as Danielson-Lanczos Lemma needs matrices with size in the power of 2
 * If the input image size is not in the power of 2, zeros-padding will be added to the size of the nearest power of 2
 */
cv::Mat reMap(cv::Mat img){
	cv::Mat image;
	int rows = img.rows;
	int cols = img.cols;
	int newrows = 0;
	int newcols = 0;
	

	if((rows & (rows -1) ) != 0){///bitwise element addition not equals 0, which is not of ^2
		//convert it to the nearest power of 2 size
		newrows = std::pow(2, ceil(log((float) rows)/log(2.0)) );
	}
	
	if((cols & (cols-1)) != 0){
		//convert it to the nearest power of 2 size
		newcols = std::pow(2, ceil(log((float) cols)/log(2.0)) );
	}

	if(newrows == 0){	newrows = rows;	}
	if(newcols == 0){	newcols = cols;	}
	
		if(img.channels() ==1){
			image = *new cv::Mat(newrows, newcols, CV_8UC1);
		}else if(img.channels() == 3){
			image = *new cv::Mat(newrows, newcols, CV_8UC3);
		}

	for(int j = 0; j < newcols; j++){
		for(int i = 0; i < newrows; i++){
			if(image.channels() == 1){
				//printf("i : %d, j : %d\n", i,j);
				if(i < img.rows && j < img.cols){
					image.at<uchar>(i,j) = img.at<uchar>(i,j);
				}else{
					image.at<uchar>(i,j) = 0;
				}
				
			}else{
				if(i < img.rows && j < img.cols){
					image.at<cv::Vec3b>(i,j) = img.at<cv::Vec3b>(i,j);
				}else{
					image.at<cv::Vec3b>(i,j)[0] = 0;
					image.at<cv::Vec3b>(i,j)[1] = 0;
					image.at<cv::Vec3b>(i,j)[2] = 0;
				}
			}
		}//end for inner loop
	}//end for loop

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
	int istep;
	//mmax is the number of times the calculation should be at maximum,
	//once the maximum amount of times to calculate goes over the data size, it stops
	while( N > mmax){//nof point transform	
		//euler's angle value (2Pi/N)
		if(inverse)	theta = two_pi/mmax;	else theta = -two_pi/mmax;		
		//next number of splits and calculations
		istep = mmax << 1;
		wtemp = std::sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = std::sin(theta);
		wr = 1.0;
		wi = 0.0;
		//half of theta


		//printf("Outer loop : mmax : %d, istep : %d\n", mmax, istep);
		//internal loop
		for(int m = 0; m < mmax; m+=2){//size of point transform
			//printf("	inner loop1 : m : %d, weight = %f\n", m, wr);
			//cv::waitKey(0);
			//get the Es and Os from the data. 
			for(int i = m; i < N-mmax; i+= istep){
				int j = i + mmax;
				//printf("		innerloop1 : i : %d, j = %d\n", i,j);
				double tempreal = wr*data->at(j);
								
				data->at(j) = (int) data->at(i) - tempreal;
				data->at(i) += (int) tempreal;
				//printf("data at j : %d, data at i : %d\n", data->at(j), data->at(i));
			}
			wr = (wr*wpr) + wr;
		}
		mmax = istep;

	}//end while
	
	//if(!inverse){
	//	for(int i = 0; i < istep; i++){
	//		data->at(i) /= istep;
	//		
	//	}
	//}

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
			//printf("x : %d, j : %d\n", x,j);
			row->at(x)  = (int) data->at<uchar>(j,x);
		}

		//std::vector<int>* temp_row = bitReversal(row);
		std::vector<int>* temp_row = FFT(row, false);
		for(int x = 0; x < width; x++){

			if(temp_row->at(x) < 0){
				temp_row->at(x) = 0;
			}
		
			data->at<uchar>(j,x) = (uchar) temp_row->at(x);
		}
	}
	
	cv::imshow("datarow", *data);
	cv::waitKey(0);

		//perform fft on each col can stuff result back into the matrix
	for(int i = 0; i < width; i++){//640
		//printf("i : %d\n", i);
		for(int y = 0; y < height; y++){//480
			col->at(y)  = (int) data->at<uchar>(y,i);
		}

		col = bitReversal(col);
		//printf("size of col %d\n", sizeof(col));
		std::vector<int>* temp_col = FFT(col, false);
		for(int j = 0; j < height; j++){	

			if(temp_col->at(j) < 0){
				temp_col->at(j) = 0;
			}
			data->at<uchar>(j,i) = (uchar) temp_col->at(j);
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
	cv::Mat img1 = cv::imread("rawdata\\setthree_with_markers\\3cframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//the reference image that is checked for a certain patch
	cv::Mat img2 = cv::imread("rawdata\\setthree_with_markers\\2cframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	img2 = reMap(img2);
	//cv::GaussianBlur(target, target9, cv::Size(21,21), 0,0);


	cv::Mat* ref = &img2;
	cv::Mat* target = &img1;
	std::vector<float> *corrs = new std::vector<float>();

	//float correlation = Norm_CrossCorr(*ref,*target,ref->cols*.5, 0, 100,100);
	//printf("correlation is %f\n", correlation);
	cv::waitKey(0);

	//get each row out and reverse them, then perform FFT
	cv::Mat* result = FFT(ref,ref->cols, ref->rows, false);
	cv::Mat* kono = FFT(result, result->cols, result->rows, true);


	//cv::waitKey(30);
	cv::imshow("result", *ref);
	cv::imshow("kono", *kono);
	cv::waitKey(0);

	
}

