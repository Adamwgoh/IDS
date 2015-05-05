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

//looks for the two markers in the frame
std::pair<cv::Rect,cv::Rect> ColorProcessor::getMarkers(cv::Mat marker, cv::Mat frame){
	//splits image into two halves that can be used to extract markers
	cv::Rect left_roi, right_roi;
	cv::Mat left_frame = cv::Mat(frame, cv::Rect(0,0, frame.cols/2, frame.rows));
	cv::Mat right_frame = cv::Mat(frame, cv::Rect((frame.cols/2)+1, 0, (frame.cols/2)-1, frame.rows));
	left_roi  = getInterestObject(marker, left_frame);
	right_roi = getInterestObject(marker, right_frame);
	right_roi = cv::Rect(right_roi.x+frame.cols/2, right_roi.y, right_roi.width, right_roi.height);

	return std::pair<cv::Rect,cv::Rect>(left_roi,right_roi);
}

//locate the roi in the object
cv::Rect ColorProcessor::getInterestObject(cv::Mat object, cv::Mat scene){
	cv::Rect roi;
	cv::Mat obj, image;
	object.copyTo(obj); scene.copyTo(image);
	//computer keypoints
	std::vector<cv::KeyPoint> object_keypoints = getInterestPoints(obj);
	std::vector<cv::KeyPoint> image_keypoints = getInterestPoints(image);

	//extract roi
	roi = MatchingKeypoints(obj, image, object_keypoints, image_keypoints);


	return roi;
}

cv::Rect ColorProcessor::MatchingKeypoints(cv::Mat img1, cv::Mat img2,std::vector<cv::KeyPoint> keypoints_1, std::vector<cv::KeyPoint> keypoints_2){
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
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	for(int i = 0; i < (int) good_matches.size(); i++){
		printf("--GoodMatch [%d] Keypoint1 : %d -- Keypoint 2: %d \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

	//extract the detected objects
	std::vector<cv::Point2f> obj, scene;

	for(int i = 0; i < good_matches.size(); i++){
		//Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}

	cv::Mat homography = cv::findHomography(obj, scene, CV_RANSAC);

	//get corners from the image
	std::vector<cv::Point2f> obj_corners(4);
	std::vector<cv::Point2f> scene_corners(4);

	//get the transform using perspective transform and extract it from the image
	obj_corners[0] = cvPoint(0,0);//topright
	obj_corners[1] = cvPoint(img1.cols,0);//bottomright
	obj_corners[2] = cvPoint(img1.cols, img1.rows);//bottomleft
	obj_corners[3] = cvPoint(0, img1.rows);//topleft

	cv::perspectiveTransform(obj_corners, scene_corners, homography);
	
	//get the largest and smallest x, largest and smallest y;
	int largestx = 0;int largesty = 0;
	int smallestx = 9999;int smallesty = 9999;
	for(int i = 0; i < scene_corners.size(); i++){
		int ex = (int) scene_corners.at(i).x; int why = (int) scene_corners.at(i).y;
		if(ex > largestx)	largestx = ex;
		if(why > largesty)	largesty = why;
		if(ex < smallestx)	smallestx = ex;
		if(why < smallesty)	smallesty = why;
	}
	if(smallestx < 0) smallestx = 0;
	if(smallesty < 0) smallesty = 0;
	if(largestx > img2.rows) largestx = img2.rows;
	if(largesty > img2.rows) largesty = img2.rows;
	int x = (int) smallestx ; int y = (int) smallesty;
	int roix = (int) smallestx + img1.cols;
	int width = largestx - smallestx;
	int height = largesty - smallesty;
	//assert that the roi is in range of image

	printf("extra width : %d, extra height : %d\n", (x+width), (x+height));
	
	if(roix+width > img_matches.cols){
		width -= ((roix+width)-img_matches.cols);

		printf("extra width\n");
	}

	if(y+height   > img_matches.rows)	height -= ((y+height)-img_matches.rows);
	printf("largestx :%d, smallestx : %d, largesty : %d, smallesty : %d\n", largestx, smallestx,  largesty, smallesty);
	printf("width :%d, height:%d\n", width, height);

	cv::Mat roi = cv::Mat(img_matches, cv::Rect(roix,y,width,height));
	
	////extract it out using the given corners
	line( img_matches, scene_corners[0] + cv::Point2f(img1.cols, 0) , scene_corners[1] + cv::Point2f(img1.cols, 0), cv::Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + cv::Point2f(img1.cols, 0), scene_corners[2] + cv::Point2f(img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + cv::Point2f(img1.cols, 0), scene_corners[3] + cv::Point2f(img1.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + cv::Point2f(img1.cols, 0), scene_corners[0] + cv::Point2f(img1.cols, 0) , cv::Scalar( 0, 255, 0), 4 );

	return cv::Rect(x,y,width,height);
}

cv::Mat ColorProcessor::getHistogram(cv::Mat src){
	cv::Mat img;
	src.copyTo(img);
	cv::Mat result;
	cv::MatND hist;
	int bins = 256;
	std::vector<int> histSize = std::vector<int>(bins);
	float range[] = {0, 255};
	const float* ranges[] = {range};
	std::vector<int> channels = std::vector<int>(0);

	cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &bins, ranges, true, false);

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
std::vector<cv::Mat> ColorProcessor::findDx(cv::Mat src, cv::Mat graph, int window_width){
	std::vector<cv::Mat> result = std::vector<cv::Mat>();
	bool foundkeyPoint = false;
	int win_x = 0;
	int ex = 0;
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
	curr_deviation = calcStandardDeviation(flat_graph, window_width/2, 0);
	
	for(int x = 1; x < flat_graph.cols-(window_width); x += window_width){

		if(x + (window_width) > flat_graph.cols-window_width){
			ex = flat_graph.cols - x;
		}

		//printf("x at : %d\n", x);
		deviation = calcStandardDeviation(flat_graph, window_width, x);
		// if there's a 10% increase in deviation, low number because window is big.
		//if(deviation != 0 && (std::abs(curr_deviation - deviation)/curr_deviation >0.5)){
		if(deviation != 0 && (deviation > 1.7*curr_deviation)){	

			//found change. get either the start of the window or end of the window
			if(!foundkeyPoint){

				//take the start of the window as the beginning of an roi
				win_x = x;
				foundkeyPoint = true;
				continue;
			}else{

				//take the end of the window as the end of an roi, and crop an area of interest
				//where two keypoints of deviance change
				cv::Mat roi = cv::Mat(graph, cv::Rect(win_x, 0, x+window_width-win_x, graph.rows));
				cv::Mat src_roi = cv::Mat(src, cv::Rect(win_x, 0, x+window_width-win_x, src.rows));
				foundkeyPoint = false;
				result.push_back(src_roi);
				continue;
				//result.push_back(roi);
			}

		}else{	curr_deviation = deviation;	}
	}

	//get the rest of the pixels as a last window to check
	deviation = calcStandardDeviation(flat_graph, flat_graph.cols-ex, ex);

	//if there's a 10% increase in deviation, low number because window is big.
	if(deviation != 0 && (deviation > 2*curr_deviation)){

		//found change. get either the start of the window or end of the window
		if(!foundkeyPoint){
			//take the start of the window as the beginning of an roi
			printf("first keypoint\n");
			foundkeyPoint = true;
		}else{
			//take the end of the window as the end of an roi, and crop an area of interest
			//where two keypoints of deviance change
			cv::Mat roi = cv::Mat(graph, cv::Rect(win_x, 0, ex, graph.rows));
			cv::Mat src_roi = cv::Mat(src, cv::Rect(win_x, 0, ex, src.rows));
			printf("second keypoint. Total window size is : %d\n", ex+window_width-win_x);
			foundkeyPoint = false;
			//result.push_back(roi);
			result.push_back(src_roi);
		}
	}

	cv::destroyWindow("roi2");
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
	for(int i = startx; i < startx+window_width; i++){
		int val = src.at<uchar>(0,i);
		double square_diff = std::pow( (double) (val - mean), 2);
		sum_squared_diff += square_diff;
	}

	//calculate its standard deviation
	double variance = sum_squared_diff/ (double) window_width;
	std_deviation = std::sqrt(variance);

	return std_deviation;
}

std::vector<cv::Vec4i> ColorProcessor::HoughLines(cv::Mat src){

	//canny it!
	cv::Mat cannysrc = cv::Mat();
	cv::Canny(src, cannysrc, 50, 200, 3);
	std::vector<cv::Vec4i> lines = std::vector<cv::Vec4i>();
	cv::HoughLinesP(cannysrc, lines, 1, CV_PI/180, 50);


	return lines;
}

//reduce color to 64
void ColorProcessor::reduceColor64(cv::Mat& src, cv::Mat& dst){
	src.copyTo(dst);

	for(int row = 0; row < src.rows; row++){
		for(int col = 0; col < src.cols; col++){
			dst.at<cv::Vec3b>(row,col).val[0] = (src.at<cv::Vec3b>(row,col).val[0])/64*124/2;
			dst.at<cv::Vec3b>(row,col).val[1] = (src.at<cv::Vec3b>(row,col).val[1])/64*124/2;
			dst.at<cv::Vec3b>(row,col).val[2] = (src.at<cv::Vec3b>(row,col).val[2])/64*124/2;
		}
	}
}

//given a raw color input image, performs k-means clustering and returns the two main classes value in Vec3b form.
//This is useful for finding the dominant colors
//when there is only two classes used
std::vector<cv::Vec3b> ColorProcessor::GetClassesColor(cv::Mat src, int k, int iteration){
	std::vector<cv::Vec3b> result_classes = std::vector<cv::Vec3b>();
	cv::Mat result = cv::Mat();

	//reshape mat into a flat row and 32bit float
	cv::Mat shaped_src(src.rows*src.cols, 3, CV_32F);
	for(int row = 0; row < src.rows; row++){
		for(int col = 0; col < src.cols; col++){
			for(int chn = 0; chn < 3; chn++){
				
				shaped_src.at<float>(row + col*src.rows, chn) = src.at<cv::Vec3b>(row,col)[chn];
			}
		}
	}

	shaped_src.convertTo(shaped_src, CV_32F);

	cv::Mat labels, centers;
	cv::kmeans(shaped_src, k,labels,cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), iteration, cv::KMEANS_PP_CENTERS, centers);

	result = cv::Mat(src.size(), src.type());

	for(int idx = 0; idx < centers.rows; idx++){
		cv::Vec3b value = cv::Vec3b();
		value[0] = centers.at<float>(idx, 0);
		value[1] = centers.at<float>(idx, 1);
		value[2] = centers.at<float>(idx, 2);
		result_classes.push_back(value);
	} 

	return result_classes;
}

//RGB delta color difference comparison. Similar to Lab's delta E formulae, it uses the RGB color space instead
//calculate the color difference
double ColorProcessor::compareColours(cv::Vec3b L_lastcolour,cv::Vec3b R_firstcolour){

	double diff = std::sqrt(	std::pow((double) L_lastcolour.val[0] - (double) R_firstcolour.val[0],2)+
								std::pow((double) L_lastcolour.val[1] - (double) R_firstcolour.val[1],2)+
								std::pow((double) L_lastcolour.val[2] - (double) R_firstcolour.val[2],2)
								);

	return diff;
}

cv::Mat ColorProcessor::ColorClusteredImg(cv::Mat src, int k, int iteration){
	cv::Mat result = cv::Mat();
	//reshape mat into a flat row and 32bit float
	cv::Mat shaped_src(src.rows*src.cols, 3, CV_32F);
	for(int row = 0; row < src.rows; row++){
		for(int col = 0; col < src.cols; col++){
			for(int chn = 0; chn < 3; chn++){
				
				shaped_src.at<float>(row + col*src.rows, chn) = src.at<cv::Vec3b>(row,col)[chn];
			}
		}
	}

	shaped_src.convertTo(shaped_src, CV_32F);

	cv::Mat labels, centers;
	cv::kmeans(shaped_src, k,labels,cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), iteration, cv::KMEANS_PP_CENTERS, centers);

	result = cv::Mat(src.size(), src.type());
	
	//get the clusters into the result matrix
	for(int row = 0; row < src.rows; row++){
		for(int col = 0; col < src.cols; col++){
			int cluster_index = labels.at<int>(row + col*src.rows, 0);
			result.at<cv::Vec3b>(row,col)[0] = centers.at<float>(cluster_index, 0);
			result.at<cv::Vec3b>(row,col)[1] = centers.at<float>(cluster_index, 1);
			result.at<cv::Vec3b>(row,col)[2] = centers.at<float>(cluster_index, 2);
		}
	}
	cv::imshow("result", result);
	cv::waitKey(40);
	cv::waitKey(0);
	cv::destroyWindow("result");

	return result;
}

cv::Mat ColorProcessor::displayColor(cv::Vec3b colorval){
	cv::Mat output = cv::Mat(50,50, CV_8UC3);

	for(int row = 0; row < output.rows; row++){
		for(int col = 0; col < output.cols; col++){
			output.at<cv::Vec3b>(row,col)[0] = colorval[0];
			output.at<cv::Vec3b>(row,col)[1] = colorval[1];
			output.at<cv::Vec3b>(row,col)[2] = colorval[2];
		}
	}

	return output;
}