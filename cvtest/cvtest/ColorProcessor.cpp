#include "ColorProcessor.h"

//constructor
ColorProcessor::ColorProcessor(){

}

//convert from RGB to HSV. HSV has values from 0-255
cv::Mat ColorProcessor::RGB2HSV(cv::Mat RGBsrc){
	cv::Mat img;
	cv::Mat result;
	RGBsrc.copyTo(img);
	result = cv::Mat(img.rows, img.cols, img.type());

	int iRed, iBlue, iGreen;
	float fRed, fBlue, fGreen;

	for(int j = 0; j < img.rows; j++){
		for(int i = 0; i < img.cols; i++){
			//get the values out and convert them to floating val
			//BGR format
			iBlue = img.at<cv::Vec3b>(j,i).val[0];
			iGreen = img.at<cv::Vec3b>(j,i).val[1];
			iRed = img.at<cv::Vec3b>(j,i).val[2];
			
			fRed = iRed / 255.0f;fGreen = iRed/255.0f; fBlue = iBlue/255.0f;
			float maximum_f, minimum_f;
			int	  maximum_int, minimum_int;
			float diff = 0;

			//find the largest and smallest val between the three rgb values;
			if(iRed > iBlue && iRed > iGreen){
				maximum_int = iRed;
				maximum_f = fRed;

				if(iBlue > iGreen){
					//Red, Blue, Green
					minimum_int = iGreen;
					minimum_f = iGreen;
				}else{
					//Red, Green, Blue
					minimum_int = iBlue;
					minimum_f = fBlue;
				}
			}else if(iBlue > iRed && iBlue > iGreen){
				maximum_int = iBlue;
				maximum_f = fRed;

				if(iRed > iGreen){
					//Blue, Red, Green
					minimum_int = iGreen;
					minimum_f = fGreen;
				}else{
					//Blue, Green, Red
					minimum_int = iRed;
					minimum_f = fRed;
				}
			}else if(iGreen > iBlue && iGreen > iRed){
				maximum_int = iGreen;
				maximum_f = fGreen;

				if(iBlue > iRed){
					//Green, Blue, Red
					minimum_int = iRed;
					minimum_f = fRed;
				}else{
					//Green, Red, Blue
					minimum_int = iBlue;
					minimum_f = fBlue;
				}
			}

			float Val, Sat, Hue;
			diff = maximum_f - minimum_f;
			Val = maximum_f;
			if(maximum_int != 0){
				
				Sat = diff / maximum_f;
				//calculate Hue
				if( maximum_int == iRed){

					Hue = (fGreen - fBlue)/((diff*6.0f));
				}else if(maximum_int == iGreen){

					Hue = (2.0f/6.0f) + ((fBlue - fRed)/(diff*6.0f));
				}else if(maximum_int == iBlue){

					Hue = (4.0f/6.0f) + ((fRed - fGreen)/(diff*6.0f));
				}
			}else{
				Hue = 0;
				Sat = 0;
			}

			//keep values within [0,1]
			if(Hue < 0.0f){	Hue += 1.0f;}else
				if(Hue > 1.0f){	Hue -= 1.0f;}


			//convert to 0-255 values
			int H = (int) (0.5f  + Hue*255.0f);
			int S = (int) (0.5f  + Sat*255.0f);
			int V = (int) (0.5f  + Val*255.0f);

			if(H > 255)	H = 255;
			if(H < 0)	H = 0;
			if(S > 255) S = 255;
			if(S < 0)	S = 0;
			if(V > 255) V = 255;
			if(V < 0)	V = 0;

			result.at<cv::Vec3b>(j,i).val[0] = H;
			result.at<cv::Vec3b>(j,i).val[1] = S;
			result.at<cv::Vec3b>(j,i).val[2] = V;
	}//end for loop
	}

	return result;
}


cv::Mat ColorProcessor::findInterestPoints(cv::Mat src){
	int minHessian = 400;

	cv::SurfFeatureDetector detector(minHessian);
	std::vector<cv::KeyPoint> keypoints;

	detector.detect( src, keypoints);
	cv::Mat imgkeypt;
	cv::drawKeypoints( src, keypoints, imgkeypt, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

	return imgkeypt;
}

std::vector<cv::KeyPoint> ColorProcessor::getInterestPoints(cv::Mat src){
	int minHessian = 400;

	cv::SurfFeatureDetector detector(minHessian);
	std::vector<cv::KeyPoint> keypoints;

	detector.detect(src, keypoints);
	
	return keypoints;
}

cv::Mat ColorProcessor::getHueImage(cv::Mat hsvsrc){
	cv::Mat result;

	std::vector<cv::Mat> split_hsv = std::vector<cv::Mat>();
	cv::split(hsvsrc, split_hsv);
	result = split_hsv[0];

	return result;
}

cv::Mat ColorProcessor::getSaturationImage(cv::Mat hsvsrc){
	cv::Mat result;

	std::vector<cv::Mat> split_hsv = std::vector<cv::Mat>();
	cv::split(hsvsrc, split_hsv);
	result = split_hsv[1];

	return result;
}

cv::Mat ColorProcessor::getValueImage(cv::Mat hsvsrc){
	cv::Mat result;

	std::vector<cv::Mat> split_hsv = std::vector<cv::Mat>();
	cv::split(hsvsrc, split_hsv);
	result = split_hsv[2];

	return result;
}

cv::Mat ColorProcessor::MatchingKeypoints(cv::Mat img1, cv::Mat img2,std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2){
		//calculate feature vectors
	cv::SurfDescriptorExtractor extractor;

	cv::Mat descriptor1, descriptor2;
	extractor.compute(img1, keypoints_1, descriptor1);
	extractor.compute(img2, keypoints_2, descriptor2);

	//Match descriptor vectors with FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches);

	double max_dist = 0;
	double min_dist = 100;

	//calculate max and min distances between keypoints
	for(int i = 0; i < descriptor1.rows; i++){
		double dist = matches[i].distance;
		if( dist > max_dist) max_dist = dist;
		if(dist < min_dist)  min_dist = dist;
	}

	//Draw only good matches, whose distance is less than 2*min_dist
	//or a small arbitary value if min_dist is too small
	std::vector<cv::DMatch> good_matches;

	for(int i = 0; i < descriptor1.rows;i++){
		if(matches[i].distance <= cv::max(2*min_dist, 0.02) ){
			good_matches.push_back(matches[i]);
		}
	}

	//Draw only good matches
	cv::Mat img_matches;
	cv::drawMatches(img1, keypoints_1, img2, keypoints_2,
		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

	for(int i = 0; i < (int) good_matches.size(); i++){
		printf("--GoodMatch [%d] Keypoint1 : %d -- Keypoint 2: %d \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

	return img_matches;
}

cv::Mat ColorProcessor::getHueHistogram(cv::Mat Huesrc){
	cv::Mat img;
	Huesrc.copyTo(img);
	cv::Mat result;
	cv::MatND hist;
	int bins = 256;
	std::vector<int> histSize = std::vector<int>(bins);
	float range[] = {0, 255}; 
	const float* ranges[] = {range};//if using openCV's cvtColor, it will be [0,180]
	std::vector<int> channels = std::vector<int>(0);

	//compute histogram
	cv::calcHist(&img, 1,0, cv::Mat(),hist,1,&bins,ranges, true, false);

	double total;
	total = img.rows*img.cols;

	result = cv::Mat(img.cols, 512, CV_8UC1, cv::Scalar(0,0,0));
	int bin_w = cvRound( (double) 512/256);

	cv::normalize(hist, hist, 0, result.rows, cv::NORM_MINMAX, -1, cv::Mat());

	for(int i = 1; i < histSize.size(); i++){
		cv::line(result, cv::Point(bin_w*(i-1), img.rows - cvRound(hist.at<float>(i-1)) ),
			cv::Point(bin_w*(i), img.rows - cvRound(hist.at<float>(i)) ),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}

	return result;
}

cv::Mat ColorProcessor::DilateRegionFilling(cv::Mat img){
	cv::Mat result = cv::Mat(img.rows, img.cols, img.type());
	cv::Mat kernel = (cv::Mat_<uchar>(3,3) << 0,1,0, 1,1,1,	0,1,0);
	
	cv::dilate(img, result, kernel);


	return result;
}

//display the image as a 1D graph
cv::Mat ColorProcessor::displayDepthGraph(cv::Mat data){
	assert(data.channels() == 1);
	cv::Mat img; data.copyTo(img);
	cv::Mat graph = cv::Mat(256, img.cols, CV_8UC1);
	double average_val = 0;
	
	for(int col = 0; col < img.cols; col++){
		for(int row = 0; row < graph.rows; row++){
			if(row < img.rows)	average_val += (int) img.at<uchar>(row,col);
			
			graph.at<uchar>(row,col) = 0;
		}
		
		average_val /= img.rows;
		graph.at<uchar>(255 - (int) average_val, col) = 255;
	}

	return graph;
}

//Search through a 1D image and look for a change in deviation. If there is, show the area of change.
//Once it found a change, it will show the area of window before change and prints its standard deviation value
//Returns the number of windows segmented based on the change of deviation. This groups areas with similar standard dev.
//together as a window
std::vector<cv::Mat> ColorProcessor::findDx(cv::Mat graph, int window_width){
	std::vector<cv::Mat> result = std::vector<cv::Mat>();
	bool foundkeyPoint = false;
	int win_x = 0;
	double deviation = 0;
	double curr_deviation = 0;
	assert(graph.cols > window_width);
	assert(graph.type() == CV_8UC1);
	//flatten the graph
	cv::Mat flat_graph = cv::Mat(1, graph.cols, CV_8UC1);

	//flatten the graph
	for(int col = 0; col < graph.cols; col++){
		for(int row = 0; row < graph.rows; row++){
			if((int) graph.at<uchar>(row,col) != 0){
				//found value, store at intensity map
				flat_graph.at<uchar>(0,col) = row;
				break;
			}
		}//end for-loop
	}

	//calculate the first deviation, just as the way planeDetector's does
	curr_deviation = calcStandardDeviation(flat_graph, window_width, 0);
	
	for(int x = 1; x < flat_graph.cols-window_width; x += window_width){
		printf("x at : %d\n", x);
		deviation = calcStandardDeviation(flat_graph, window_width, x);
		cv::Mat roi2 = cv::Mat(graph, cv::Rect(x, 0, window_width, graph.rows));
	
		cv::imshow("roi2", roi2);
		cv::waitKey(40);
		cv::waitKey(0);
		// if there's a 10% increase in deviation, low number because window is big.
		if(deviation != 0 && (std::abs(curr_deviation - deviation) > curr_deviation*0.3)){
			//found change. get either the start of the window or end of the window
			printf("found change in deviation\n");		
			printf("curr deviation %f, ", curr_deviation);
			printf("deviation %f\n", deviation);

			if(!foundkeyPoint){
				//take the start of the window as the beginning of an roi
				win_x = x;
				foundkeyPoint = true;
			}else{
				//take the end of the window as the end of an roi, and crop an area of interest
				//where two keypoints of deviance change
				cv::Mat roi = cv::Mat(graph, cv::Rect(win_x, 0, x-win_x, graph.rows));
				//cv::imshow("found roi", roi);
				cv::waitKey(40);
				//cv::waitKey(0);
				result.push_back(roi);
			}

		}else{

			curr_deviation = deviation;
		}
	}

	return result;
}

double ColorProcessor::calcStandardDeviation(cv::Mat src, int window_width, int startx){
	assert((startx >= 0 && startx < src.cols));
	assert(src.type() == CV_8UC1);

	//calculate mean in the window
	double mean = 0;
	double sum_squared_diff = 0;
	double std_deviation = 0;

	for(int x = startx; x < startx+window_width; x++){

		double value = (double) src.at<uchar>(0,x);
		mean += (double) src.at<uchar>(0, x);
	}
	mean /= window_width;

	//calculate its square-differences
	for(int i = 0; i < src.cols; i++){
		int val = src.at<uchar>(0,i);
		double square_diff = std::pow( (double) (val - mean), 2);
		sum_squared_diff += square_diff;
	}

	//calculate its standard deviation
	double variance = sum_squared_diff/ (double) src.cols-1;
	std_deviation = std::sqrt(variance);

	return std_deviation;
}