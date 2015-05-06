#include "depthPlaneDetector.h"
#include "stdafx.h"

const int VGA_WIDTH = 640;
const int VGA_HEIGHT = 480;
const int QQVGA_WIDTH = 160;
const int QQVGA_HEIGHT = 120;


//empty constructor
depthPlaneDetector::depthPlaneDetector(){
	
	excerpts = std::vector<cv::Rect>();
}

depthPlaneDetector::depthPlaneDetector(cv::Mat* input, int w_ksize, int h_ksize)
: image(input), windowsize(cv::Size(w_ksize,h_ksize) )
{
	excerpt_xs = std::vector<int>();
	excerpts = std::vector<cv::Rect>();
	//printf("Initialized depthPlaneDetector\n");
	//printf("Window size : (%d,%d)\n", w_ksize, h_ksize);
}

depthPlaneDetector::depthPlaneDetector(int w_ksize, int h_ksize)
	: image(&cv::Mat()), windowsize(cv::Size(w_ksize,h_ksize) )
{
	excerpt_xs = std::vector<int>();
	excerpts = std::vector<cv::Rect>();
	//printf("Initialized depthPlaneDetector.\n");
	//printf("Window size : (%d,%d)\n", w_ksize, h_ksize);
}

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
	
	//calculate mean in the window area
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

//search for a drop in the depth graph and captures it.
std::vector<cv::Rect> depthPlaneDetector::searchDepthDeviation(cv::Mat depth_img, cv::Mat depth_graph){
	std::vector<cv::Rect> result = std::vector<cv::Rect>();
	double prev_deviation = 0;
	double curr_deviation = 0;
	//cv::Mat deviation_graph = cv::Mat::zeros(1, depth_img.cols, CV_64FC1);
	int start_y = (depth_img.rows/2)+(windowsize.height);//put window in the middle for least noise
	prev_deviation = calcStandardDeviation(&depth_img, windowsize,0,start_y);

	//get the standard deviation across the graph as the window slides across
	for(int start_x = 1; start_x < depth_img.cols-windowsize.width; start_x+=windowsize.width/2){
			
			curr_deviation = calcStandardDeviation(&depth_img, windowsize,start_x,start_y);
			//printf("curr_deviation : %f, deviation : %f\n", curr_deviation, prev_deviation);
			if(curr_deviation >= prev_deviation*4 && !(curr_deviation == prev_deviation && curr_deviation == 0)){
				//found roi
				result.push_back(cv::Rect(start_x, 0, windowsize.width, depth_graph.rows));
				cv::Mat roi = cv::Mat(depth_graph, cv::Rect(start_x, 0, windowsize.width, depth_graph.rows));
				cv::Mat prev_roi = cv::Mat(depth_graph, cv::Rect(start_x-(windowsize.width/2),0,windowsize.width, depth_graph.rows));
			}
			prev_deviation = curr_deviation;

			//deviation_graph.at<double>(0,start_x+(windowsize.width/2)) = (double) curr_deviation;

	}//end x-loop


	return result;
}

/**
 *	Search through the input image for a change in deviation. If there is, show the area where deviation is found
 **/
std::vector<cv::Mat> depthPlaneDetector::searchDeviationDx(cv::Mat input, cv::Mat colorimg){
	std::vector<cv::Mat> result = std::vector<cv::Mat>();
	double deviation = 0;
	double curr_deviation = 0;
	cv::Mat deviation_graph = cv::Mat::zeros(1, input.cols, CV_64FC1);
	assert(input.rows > windowsize.height && input.cols > windowsize.width);
	int start_y = (input.rows/2)+(windowsize.height);//put window in the middle for least noise
	//calculate the first deviation, this is to make sure any changes at the start of the window can be detected
	curr_deviation = calcStandardDeviation(&input, windowsize, 0, start_y);

	//get the standard deviation across the graph as the window slides across
	for(int start_x = 0; start_x < input.cols-windowsize.width; start_x++){
			
			curr_deviation = calcStandardDeviation(&input, windowsize,start_x,start_y);
			deviation_graph.at<double>(0,start_x+(windowsize.width/2)) = (double) curr_deviation;
			//printf("deviation_graph val : %f\n", deviation_graph.at<double>(0, start_x+(windowsize.width/2)) );

	}//end x-loop
	cv::Mat devgraph = displayDepthGraph(deviation_graph,0,0);
	//cv::imshow("devgraph", devgraph);
	//cv::waitKey(30);
	//cv::waitKey(0);
	//get keypoint in image based on manual thresholding
	result = findEdges(deviation_graph, colorimg);
	//printf("result size : %d\n", result.size());
	//cv::imshow("deviation_graph", deviation_graph);
	//cv::waitKey(40);
	//cv::waitKey(0);
	cv::destroyAllWindows();

	return result;

}

std::vector<cv::Mat> depthPlaneDetector::findEdges(cv::Mat devgraph, cv::Mat colorimg, double max, std::vector<int> max_x){
	//printf("finding edges..\n");
	std::vector<cv::Mat> result = std::vector<cv::Mat>();
	std::vector<int> pos_xs = std::vector<int>();
	double hill_thresh;
	cv::Mat graph =  displayGraph(devgraph);

	//set the threshold as 20% above the minimum
	double average_val = 0;
	double max_val = 0;int max_col = 0;
	double largest = 0; if(max != 0.0f)	largest = max;

	//get the average value used to set threshold
	for(int col = (windowsize.width/2)+1; col < devgraph.cols-(windowsize.width/2); col++){
		average_val += devgraph.at<double>(0, col);
	}
	average_val /= devgraph.cols - windowsize.width+1;
	
	//threshold is 50% above average value
	hill_thresh = (double)average_val*1.5;
	int startx = 0;int endx = 0;bool withinhills = false;
	//look for any hills above the threshold. save positions of the peak of those hills
	for(int col = (windowsize.width/2) +1; col < devgraph.cols-windowsize.width; col++){
		
		double val = devgraph.at<double>(0, col);
		//get the highest val
		if(val >= hill_thresh && val >= max_val){
			//if value is the largest but smaller than the largest hill before and not within the windowed area
			if(val < largest && !max_x.empty()){
				//check if value is within the range of any found hills
				for(int k = 0; k < max_x.size();k++){
					if((col > max_x.at(k)-windowsize.width || col < max_x.at(k)+windowsize.width)){
						withinhills = true;
					}	
				}

				if(col+windowsize.width < devgraph.cols && !withinhills){
					//printf("max_val not within max_x %d is %f, at %d\n", max_x, val, col);
					max_val = val;
					max_col = col;
				}
		
			}else if( largest == 0){
				//no limit/first run
				//printf("max_val is now :%f at col :%d\n", max_val, max_col);
				max_val = val;
				max_col = col;				
			}
		}
	}//end for loop
	//printf("max val : %f, max_col : %d\n", max_val, max_col);
	//create an roi region for it
	if(max_col == 0){
		//no roi found
		//printf("no roi found\n");
		return result;
	}

	//printf("found one roi\n");
	cv::Mat roi(colorimg, cv::Rect(max_col, 0, windowsize.width,graph.rows));

	result.push_back(roi);
	max_col = convertOffset(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(QQVGA_WIDTH,QQVGA_HEIGHT), max_col, 0).first;
	cv::Size winsize = convertSize(cv::Size(VGA_WIDTH,VGA_HEIGHT), cv::Size(QQVGA_WIDTH,QQVGA_HEIGHT), windowsize.width, windowsize.height);
	excerpts.push_back( cv::Rect(max_col, 0, winsize.width,QQVGA_HEIGHT));
	excerpt_xs.push_back(max_col);

	std::vector<cv::Mat> temp = std::vector<cv::Mat>();
	std::vector<int> max_cols = max_x;
	max_x.push_back(max_col);
	temp = findEdges(devgraph, colorimg, max_val, max_cols);
	while(!temp.empty()){
		cv::Mat mat;
		for(int k = 0; k < temp.size();k++){
			cv::Mat mat = temp.at(k);
			result.push_back(mat);
		}
	}

	return result;
}

//converts the offset coordinates in the first image to the offset coordinates in the second offset
std::pair<int, int>	depthPlaneDetector::convertOffset(cv::Size src, cv::Size targ, int offsetx, int offsety){
	//find scale factor between two sizes
	double width_ratio = (double) targ.width/ (double) src.width;
	double height_ratio = (double) targ.height/(double) src.height;

	int new_offx = std::floor(offsetx*width_ratio);
	int new_offy = std::floor(offsety*height_ratio);
	std::pair<int,int> new_off = std::pair<int, int>(new_offx, new_offy);

	return new_off;
}

cv::Size depthPlaneDetector::convertSize(cv::Size src, cv::Size targ, int width, int height){
	//find scale factor between two sizes
	double width_ratio = (double) targ.width/ (double) src.width;
	double height_ratio = (double) targ.height/(double) src.height;

	int new_width = std::floor(width*width_ratio);
	int new_height = std::floor(height*height_ratio);

	return cv::Size(new_width, new_height);
}

cv::Mat depthPlaneDetector::displayGraph(cv::Mat data){
	assert(data.channels() == 1);

	cv::Mat rawdata = cv::Mat();
	data.copyTo(rawdata);
	
	cv::Mat* graph = new cv::Mat(256, rawdata.cols, CV_64FC1);
	double average_val = 0;

	for(int col = 0; col < rawdata.cols; col++){
		for(int row = 0; row < rawdata.rows; row++){
			if(row < rawdata.rows){
				average_val += (double) rawdata.at<double>(row,col);
			}
			graph->at<double>(row, col) = 0;
		}

		average_val /= rawdata.rows;
		graph->at<double>(255 - average_val, col) = 255;
		average_val = 0;
	}

	return *graph;
}

cv::Mat depthPlaneDetector::displayDepthGraph(cv::Mat data, int startx, int starty){
	assert(data.channels() == 1);

	cv::Mat rawdata = data;
	rawdata.convertTo(rawdata, CV_8UC1);
	cv::Mat* graph = new cv::Mat(256, rawdata.cols, CV_8UC1);
	double average_val = 0;
	
		for(int col = startx; col < rawdata.cols; col++){
			for (int row = starty; row < graph->rows; row++){
				if(row < rawdata.rows){

					average_val += (int) rawdata.at<uchar>(row,col);
				}
				graph->at<uchar>(row,col) = 0;
			}//endof inner-for

			average_val /= ( (rawdata.rows) );
			graph->at<uchar>(255-(int) average_val, col) = 255;
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

std::vector<cv::Rect> depthPlaneDetector::getExcerptWindow(){
	return excerpts;
}

Line::Line(cv::Mat data, bool isRANSAC, int nof_iteration, int Error_thresh):
points(data), nof_iter(nof_iteration),err_thresh(Error_thresh) {
	//extract data form given matrix
	data.copyTo(points); //one can never get too careful
	
	assert(points.type() == CV_8UC1);
	xs = std::vector<double>();
	ys = std::vector<double>();

	for(int col = 0; col < points.cols; col++){
		for(int row = 0; row < points.rows; row++){
			if(points.at<uchar>(row,col) != 0){
				//`printf("row : %d\n", row);
				xs.push_back(col);
				ys.push_back(row);
			}
		}
	}

	//perform RANSAC
	double err = RANSAC(xs, ys);
}

//default constructor
Line::Line(){
}

cv::Mat Line::drawLine(cv::Mat src, Line line){

	cv::Mat result = cv::Mat();
	src.copyTo(result);
	//choose a point at the start and the end and calculate the y coord
	int x0,y0,x1,y1;
	x0 = 0;
	y0 = x0*line.gradient + line.yintercept;
	x1 = src.cols-1;
	y1 = x1*line.gradient + line.yintercept;

	cv::line(result, cv::Point(x1,y1), cv::Point(x0,y0), cv::Scalar(180,180,180));
	
	return result;
}



cv::Mat Line::drawLine(cv::Mat src, Line line, cv::Rect area){
	cv::Mat result = cv::Mat();
	src.copyTo(result);
	cv::Mat roi = cv::Mat(result, area);
	cv::Mat drawnline = line.drawLine(roi, line);
	drawnline.copyTo(roi);
	//choose a point at the start and the end and calculate the y coord
	int x0,y0,x1,y1;
	x0 = area.x;
	y0 = x0*line.gradient + line.yintercept;

	x1 = area.x+area.width;
	y1 = x1*line.gradient + line.yintercept;
	
	while(y1 >= result.rows){
		x1--;
		y1 = x1*line.gradient + line.yintercept;
	}
	cv::line(result, cv::Point(x1,y1), cv::Point(x0,y0), cv::Scalar(180,180,180));

	return result;
}

Line::Line(cv::Mat data){

	//extract data from given matrix
	gradient = 0;
	yintercept = 0;
	data.copyTo(points);
	assert(points.type() == CV_8UC1);
	std::vector<int> ydata = std::vector<int>();
	double x  = 0;
	double y  = 0;
	double xy = 0;
	double x2 = 0;


	for(int col = 0; col < points.cols; col++){
		for(int row = 0; row < points.rows; row++){
			if(points.at<uchar>(row,col) != 0){
				//printf("row : %d\n", row);
				ydata.push_back(row);
				x += col;
				y += row;
				xy += col*row;
				x2 += col*col;
			}
		}
	}//end for
	
	gradient = (ydata.size()*xy - x*y)/(ydata.size()*x2-x*x);
	yintercept = (y - gradient*x)/ydata.size();

	printf("line model initialised with given data\n");
}

//calculate the gradient parameters. if RANSAC input parameter is true, it considers only dataset set as true as inliers
double Line::calcGradient(bool only_inliers){
	double grad;
	assert(!points.empty());
	assert(points.rows > 0 && points.cols > 0);
	assert(points.channels() == 1);//accept only 2d datapoints
	
	calcMean(xs,ys, only_inliers);
	calcstdDeviation(xs, ys, only_inliers);
	calcCorrelation(xs, ys, only_inliers);
	double dev_diff = (std_deviationx/std_deviationy);
	if(std_deviationy+0 == 0){
		dev_diff = 0;
	}
	grad = correlation*dev_diff;

	gradient = grad;
	return grad;
}



double Line::calcYintercept(bool only_inliers){
	double intercept;
	
	assert(xs.size() > 0 && ys.size() > 0 && xs.size() == ys.size());
	double grad = gradient;
	double mean_x = meanx;
	calcMean(xs, ys, only_inliers);
	calcGradient(only_inliers);
	printf("correlation : %f, std_devx: %f, std_devy : %f\n", correlation, std_deviationx, std_deviationy);
	//assert(grad == gradient);
	intercept = meany - gradient*meanx;

	yintercept = intercept;
	return intercept;
}

void Line::calcMean(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers){
	double sumx = 0;
	double sumy = 0;
	double inliersize = 0;

	assert(points_x.size() == points_y.size());

	for(int i = 0; i < xs.size(); i++){
		if(only_inliers){
			if(inliers.at(i)){
				inliersize++;
				sumx += points_x.at(i);
				sumy += points_y.at(i);
			}
		}else{
			sumx += points_x.at(i);
			sumy += points_y.at(i);
		}
	}

	if(only_inliers){
		if(inliersize+0 == 0){
			meanx = 0;
			meany = 0;
		}else{
			meanx = sumx/inliersize;
			meany = sumy/inliersize;
		}
		
	}else{
		meanx = sumx/(double) points_x.size();
		meany = sumy/(double) points_y.size();
	}
}

void Line::calcstdDeviation(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers){

	double mean_x = meanx;
	double mean_y = meany;
	double sum_squared_diffx, sum_squared_diffy;
	double inliersize = 0;
	assert(points_x.size() > 0 && points_y.size() > 0 && points_x.size() == points_y.size());
	
	for(int i = 0; i < points_x.size(); i++){
		if(only_inliers){
			if(inliers.at(i)){
				inliersize++;
				double square_diffx = std::pow(points_x.at(i) - mean_x, 2);
				double square_diffy = std::pow(points_y.at(i) - mean_y, 2);

				sum_squared_diffx += square_diffx;
				sum_squared_diffy += square_diffy;
			}
		}else{
			double square_diffx = std::pow(points_x.at(i) - mean_x, 2);
			double square_diffy = std::pow(points_y.at(i) - mean_y, 2);

			sum_squared_diffx += square_diffx;
			sum_squared_diffy += square_diffy;
		}
	}

	double variancex = 0; double variancey = 0;
	if(only_inliers){
		//calculate the std deviation for both variables
		if(inliersize+0 == 0){
			printf("no inliers at all\n");
			variancex = 0;
			variancey = 0;
		}else{
			variancex = sum_squared_diffx/inliersize;
			variancey = sum_squared_diffy/inliersize;
		}
	}else{
		//calculate the std deviation for both variables
		variancex = sum_squared_diffx/points_x.size();
		variancey = sum_squared_diffy/points_x.size();
	}
	if(std_deviationx < 0.0000000000001 && std_deviationy < 0.00000000000001){
		printf("hold on sec\n");
	}
	std_deviationx = std::sqrt(variancex);
	std_deviationy = std::sqrt(variancey);
}

void Line::calcCorrelation(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers){

	double mean_x = meanx;
	double mean_y = meany;
	int trues = 0;
	double xy = 0;double x2 = 0; double y2 = 0;
	double sumx = 0; double sumy = 0;
	double corr = 0;
	assert(xs.size() > 0 && ys.size() > 0 && xs.size() == ys.size());

	for(int i = 0; i < xs.size(); i++){
		if(only_inliers){
			if(inliers.at(i)){
				trues++;
				//printf("inlier : %d y is : %f\n", i, ys.at(i));
				sumx += xs.at(i);sumy += ys.at(i);
				
				xy += (xs.at(i))*(ys.at(i));
				x2 += std::pow((xs.at(i)) ,2);
				y2 += std::pow((ys.at(i)) ,2); 
			}

		}else{
			sumx += xs.at(i);sumy += ys.at(i);
				
			xy += (xs.at(i) - meanx)*(ys.at(i) - meany);
			x2 += std::pow((xs.at(i) - meanx) ,2);
			y2 += std::pow((ys.at(i) - meany) ,2); 
		}
	}

	if(only_inliers){
		printf("(510) depthPlaneDetector : nof inliers %d\n", trues);
		double denom = std::sqrt((trues*(x2) - std::pow( sumx, 2))*(trues*y2 - std::pow(sumy,2)));
		double numerator = (trues*(xy) - (sumx*sumy));
		corr = numerator/denom;
	}else{
	
		double denom = std::sqrt((xs.size()*(x2) - std::pow( sumx, 2))*(xs.size()*y2 - std::pow(sumy,2)));
		double numerator = (trues*(xy) - (sumx*sumy));
		corr = numerator/denom;
	}
	
	correlation = corr;
}

//Line error is calculated of root mean square error
double Line::calcLineErr(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers){
	double err;
	double square_error = 0;
	double inliersize = 0;

	assert(points_x.size() > 0 && points_y.size() > 0 && points_x.size() == points_y.size());
	for(int i = 0; i <points_x.size(); i++){
			if(only_inliers){
				if(inliers.at(i)){
					inliersize++;
				}
			}	
	
		double temp_y = gradient*points_x.at(i) + yintercept;
		square_error += std::pow(points_y.at(i) - temp_y,2);
	}

	if(only_inliers){
		err = std::sqrt(square_error/inliersize);
		return err;
	}else{
		err = std::sqrt(square_error/points_x.size());
		return err;
	}
}

double Line::RANSAC(std::vector<double> points_x, std::vector<double> points_y){
	double bestErr = 9999999;
	double grad, intercept, err;
	double inliersize = 0;
	double largest_insize = 0;

	for(int i = 0; i < nof_iter; i++){
		printf("iteration no : %d\n", i);
		//select two random points from data
		inliersize = 0;
		int size = points_x.size();
		int rand1 = rand() % size;
		int rand2 = rand() % size;
		while(rand2 == rand1){	rand2 = rand()%size;	}

		double x1 = points_x.at(rand1);
		double y1 = points_y.at(rand1); 
		double x2 = points_x.at(rand2);
		double y2 = points_y.at(rand2); 

		//calculate the initial parameters with these two points
		gradient	= (y1-y2)/(x1-x2);
		yintercept	= y1 - gradient*x1;
		inliers = std::vector<bool>(points_x.size());

		//with the given parameters, determine the inliers from the dataset
		for(int ptx = 0; ptx < points_x.size(); ptx++){
			int trues = 0;
			double temp_y = gradient*points_x.at(ptx) + yintercept;
			double temp_x;

			temp_x = (points_y.at(ptx) - yintercept)/gradient;
			if(gradient+0 == 0){
				temp_x = 0;
			}

			double differences = std::max(std::abs(points_y.at(ptx) - temp_y), std::abs(points_x.at(ptx) - temp_x));
			if(differences < err_thresh){
				inliersize++;
				inliers.at(ptx) = true;trues++;
			}else{
				inliers.at(ptx) = false;	
			}
			
			//printf("trues : %d\n", trues);
		}

		//with the given inliers, calculate the root mean square error and new parameters
		grad = calcGradient(true); intercept = calcYintercept(true); err = calcLineErr(points_x, points_y, true);

		//recalculate inliers with the new parameters
		inliersize = 0;
		for(int pts = 0; pts < points_x.size(); pts++){
			double temp_y = gradient*points_x.at(pts) + yintercept;
			double temp_x = (points_y.at(pts) - yintercept)/gradient;

			double differences = std::max(std::abs(points_y.at(pts) - temp_y), std::abs(points_x.at(pts) - temp_x));
			if(differences < err_thresh){
	
				inliers.at(pts) = true;
				inliersize++;
			}else{	inliers.at(pts) = false;	}
		
		}

		//calculate new root meam square error again
		calcGradient(true); calcYintercept(true); err = calcLineErr(points_x, points_y, true);


		//is the new error minimum, if yes, assign its parameters
		if(largest_insize <= inliersize){
			largest_insize = inliersize;	
			grad = gradient;
			intercept = yintercept;
			bestErr = err;
		}

		//exit if exit condition checks
		double inlierration = (double) largest_insize/points_x.size();
		printf("current model Error %f with inlier ratio : %f, nof inlier : %d\n", err, inlierration, inliersize);
		if( err < 0.1f && inlierration > 0.7f){
			printf("EXIT CONDITION MET\n");
			cv::waitKey(0);
			break;
		}

		inliersize = 0;
		//recreate inliers with the new parameters
		for(int k = 0; k < points_x.size(); k++){
			double temp_y = grad*points_x.at(k) + intercept;
			double temp_x = (points_y.at(k) - intercept)/grad;
			double differences = std::max(std::abs(points_y.at(k) - temp_y), std::abs(points_x.at(k) - temp_x));
			
			if(differences < err_thresh){
				inliersize++;
				inliers.at(k) = true;
			}else{	inliers.at(k) = false;	}
		}
	}

	return bestErr;
}

//inputs an roi and returns the number of lines that can be generated from it
std::vector<Line> depthPlaneDetector::piecewiseLinearRegression(cv::Mat graphroi){
	std::vector<Line> lines = std::vector<Line>();

	//first minimise the area where breakpoint might be at.
	//This is done by finding a great gradient change.
	double curr_grad, grad;
	int ex,begin;
	cv::Size win(graphroi.cols/4, graphroi.rows/2);
	Line initline(cv::Mat(graphroi, cv::Rect(0,0,win.width,graphroi.rows)), true, 20, 100.0f);
	curr_grad = initline.gradient;
	for(int startx = 0; startx < graphroi.cols-win.width; startx++){

		if(startx + (win.width) > graphroi.cols-win.width){
			begin = startx;
			ex = graphroi.cols - startx;
		}

		cv::Mat roi = cv::Mat(graphroi, cv::Rect(startx, 0, win.width, graphroi.rows));
		Line aline(roi);
		aline.drawLine(roi, aline);
		grad = aline.gradient;
		if(grad != curr_grad){
			printf("gradient change!\n");
			printf("grad val : %f\n", grad);
			aline.drawLine(roi, aline);
			curr_grad = grad;
			cv::waitKey(40);
		}
	}

	//take the rest of the pixels for a final check
	//cv::Mat last_roi = cv::Mat(graphroi, cv::Rect(begin, 0, ex, graphroi.rows));
	//Line lastline(last_roi, true, 20, 100.0f);
	//lastline.drawLine(last_roi, lastline);
	//double gra = lastline.gradient;
	//if(gra > curr_grad*1.5){
	//	printf("gradient change!\n");
	//	cv::imshow("roi", last_roi);
	//	cv::waitKey(40);
	//	cv::waitKey(0);
	//}

	return lines;
}