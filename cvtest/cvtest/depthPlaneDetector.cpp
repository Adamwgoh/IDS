#include "depthPlaneDetector.h"

depthPlaneDetector::depthPlaneDetector(cv::Mat* input, int w_ksize, int h_ksize)
: image(input), windowsize(cv::Size(w_ksize,h_ksize) )
{
	printf("Initialized depthPlaneDetector\n");
}
//
//int main(int argc, char* argv[]){
//	clock_t timer;
//	timer = clock();
//	//target image is the one not moving
//	cv::Mat img1 = cv::imread("rawdata\\test\\left_testframe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//good example of noise (11depthframe)
//	cv::Mat depth1 = cv::imread("rawdata\\setthree_with_markers\\5depthframe.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	//the reference image that is checked for a certain patch
//	cv::Mat img2 = cv::imread("rawdata\\test\\right_testframe.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//
//	cv::GaussianBlur(img1, img1, cv::Size(21,21), 0, 0);
//	cv::GaussianBlur(img2, img2, cv::Size(21,21), 0,0);
//	cv::medianBlur(depth1, depth1, 3);
//	cv::Mat* target = &img2;
//	cv::Mat* ref = &img1;
//
//	depthPlaneDetector planeDetect = depthPlaneDetector(&depth1, 40, 20);
//	cv::imshow("original frame", depth1);
//	cv::waitKey(30);
//	cv::Mat deviation = planeDetect.searchDeviationDx(depth1);
//	timer = clock() - timer;
//	
//	if(!deviation.empty()){	
//		cv::imshow("deviation window", deviation);
//	}else{
//		printf("No deviation found!\n");
//	}
//
//	cv::Mat depthgraph = planeDetect.displayDepthGraph(depth1, 0, 0);
//	planeDetect.saveDepthFrame("depthgraph",&depthgraph,5);
//
//	cv::imshow("depthgraph", depthgraph);
//	cv::waitKey(30);
//	//cv::waitKey(0);
//	printf("window begin at coord : (%d,%d)\n",planeDetect.x,planeDetect.y);
//	printf("total runtime : %f\n", ((double) timer/(double) CLOCKS_PER_SEC));
//	cv::waitKey(0);
//
//	return 0;
//}

cv::Size depthPlaneDetector::getWindowsize(){	return windowsize;	}
cv::Mat depthPlaneDetector::getReferenceImage(){	return *image;	}
void depthPlaneDetector::setWindowsize(cv::Size ksize){	windowsize = ksize;	}



/**
 *	Calculates the standard deviation for the input image for a certain window
 *	at a certain location
 **/
double depthPlaneDetector::calcStandardDeviation(cv::Mat* input, cv::Size ksize, int startx, int starty){
	
	assert((startx >= 0 && startx < input->cols) && (starty >= 0 && starty < input->rows) );
	assert(input->channels() == 1);
	double totalsize, variance;
	//get ROI
	cv::Rect roi = cv::Rect(startx, starty, ksize.width, ksize.height);
	cv::Mat window = cv::Mat(*input, roi);
	window.convertTo(window, CV_8UC1);
	assert(window.channels() == 1);
	
	//display Image for test
	//cv::imshow("region of interest", window);

	//calculate mean in the window area
	cv::Mat deviance = cv::Mat(window.rows, window.cols, CV_8UC1);
	double mean = 0;
	double sum_squared_diff = 0;
	double std_deviation = 0;
	for(int x=0; x <(window.cols); x++){
		for(int y=0; y < (window.rows); y++){

			mean += window.at<uchar>(y,x);
		}
	}
	mean /= (window.cols*window.rows);
	//transform values to its deviance value
	for(int x=0; x < window.cols; x++){
		for(int y=0; y < window.rows; y++){
				double square_diff = std::pow( (window.at<uchar>(y,x) - mean) ,2);
				sum_squared_diff += square_diff;
		}
	}

	//calculate its standard deviation
	totalsize = (input->rows*input->cols) -1;
	//printf("variance : %f, sum_squared_diff : %f, totalsize : %f\n", variance, sum_squared_diff, totalsize);

	variance = sum_squared_diff/(double) totalsize;
	
	std_deviation = std::sqrt(variance);

	return std_deviation;
}

/**
 *	Search through the input image for a change in deviation. If there is, show the area where deviation is found
 **/
cv::Mat depthPlaneDetector::searchDeviationDx(cv::Mat input){
	cv::Mat result;
	double deviation = 0;
	double curr_deviation = 0;
	assert(input.rows > windowsize.height && input.cols > windowsize.width);
	int start_y = (input.rows/2)+(windowsize.height);
	for(int start_x = 0; start_x < input.cols-windowsize.width; start_x += windowsize.width/2){
			
			curr_deviation = calcStandardDeviation(&input, windowsize,start_x,start_y);
			//cv::waitKey(100);

			printf("curr_deviation : %f, deviation : %f\n", curr_deviation, deviation);
			if(deviation != 0 && (2*deviation < curr_deviation)){

				//found change, return window
				cv::Rect roi = cv::Rect(start_x,start_y,windowsize.width, windowsize.height);
				result = cv::Mat(input,roi);
				x = start_x; y = start_y;
				return result;
			}else{
				x = 0; y = 0;
				deviation = curr_deviation;
			}
	}//end x-loop
	
	return result;
}

cv::Mat depthPlaneDetector::displayDepthGraph(cv::Mat data, int startx, int starty){
	assert(data.channels() == 1);

	cv::Mat rawdata = data;
	rawdata.convertTo(rawdata, CV_8UC1);
	cv::Mat* graph = new cv::Mat(255, rawdata.cols, CV_8UC1);
	double average_val = 0;
	//used for ignoring noise values
	int reducedrows = rawdata.rows;
	
		for(int col = startx; col < rawdata.cols; col++){
			for (int row = starty; row < graph->rows; row++){
				if(row < rawdata.rows){

					average_val += (int) rawdata.at<uchar>(row,col);
				}
				graph->at<uchar>(row,col) = 0;
			}//endof inner-for

			average_val /= ( (rawdata.rows) );
			graph->at<uchar>((int) average_val, col) = 255;
			average_val = 0;
		}//endof outer-for

	return *graph;
}

//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension
void depthPlaneDetector::saveDepthFrame(const char* filename, cv::Mat* image, int framecount){
	std::ostringstream stream;
	stream << "rawdata\\";
	stream << framecount;
	stream << filename;
	stream << ".jpg";
	cv::String dframe = stream.str();
	cv::imwrite(dframe, *image);
}

//Fits a polynomial to all pixels in a 1D intensity image.
//Returns a Mat with the polynomial line drawn on the original 1D intensity image.
cv::Mat depthPlaneDetector::drawPolynomial(cv::Mat graph){
	cv::Size window = windowsize;
	bool found = false;
	
	cv::Mat srcgraph = cv::Mat();
	graph.copyTo(srcgraph);
	srcgraph.convertTo(srcgraph ,CV_8UC1);
	cv::Mat win_intensity = cv::Mat::zeros(1, srcgraph.cols, CV_8UC1);
	cv::Mat result = cv::Mat::zeros(256, srcgraph.cols, CV_8UC1);

	
	//flatten the graph matrix, since there's no need for visualization
	//gets the intensity value at each columns and store it in a flat intensity matrix
	for(int j = 0; j < srcgraph.cols; j++){
		for(int i = 0; i < srcgraph.rows; i++){
			if((int) srcgraph.at<uchar>(i,j) != 0){
				//found value, store at intensity map
				win_intensity.at<uchar>(0,j) = i;
				break;
			}
		}
	}

	//slide window across to calculate value at window's middle
	//puts value at a result Mat
	for(int j = 0; j < srcgraph.cols-window.width; j++){
		//Assign the window's least fit error to the center of window's pixel, generating a line
		int rowval = getLeastSquare(win_intensity, j, window);
		if(rowval > 255){
			rowval = 255;
		}
		result.at<uchar>(rowval,j+(int) ((window.width)-1/2)) = 255;
	}

	return result;
}

/**
 *	Gets the 1st order least square error for the window
 **/
int depthPlaneDetector::getLeastSquare(cv::Mat src, int win_x, cv::Size window){
	int leastsquare = 0;
	int x = -((window.width-1) /2);
	cv::Mat win_intensity = cv::Mat(1, window.width, CV_8UC1);
	assert(win_x < src.cols-window.width);
	assert(window.width % 2 == 1);
	
	int dx_atzero  =  (int) src.at<uchar>(0,win_x+(int) ((window.width-1)/2)) -
		(int) src.at<uchar>(0,win_x+1+(int) ((window.width-1)/2)); 
	int val_atzero = (int) src.at<uchar>(0,win_x+(int) ((window.width-1)/2));
	int dx_atone = (int) src.at<uchar>(0,win_x+1+(int) ((window.width-1)/2));

	//accumulate each ||Ii - (I0 + i(dx(0)))||^2, or the least square errors for each x
	for(int i = win_x; i < win_x+window.width; i++){
		int x_val = val_atzero - (val_atzero + x*(dx_atzero));
		x_val = (int) std::pow((double) x_val, 2);
		leastsquare += x_val;
		x++;
	}
	leastsquare = (int) std::sqrt((double) leastsquare);

	return leastsquare;
}

