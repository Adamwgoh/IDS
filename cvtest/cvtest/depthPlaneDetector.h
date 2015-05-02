#pragma once
#include "stdafx.h"


//Line describes the charactersistics of the line in a 1D image analysis.
//Line stores basic parameter values of the simple line model : y = mx + b
//Note that data used to find a line in has no values in the specific (x,y) coord, as it is
//merely a plot that shows the relationship between x and y
class Line{
//public variables
public:
	double gradient;
	double yintercept;
public:
	//constructor
	Line();
	Line(double grad, double intercept);
	Line(cv::Mat data);
	Line::Line(cv::Mat data, bool isRANSAC, int nof_iteration, int Error_thresh);

	cv::Mat drawLine(cv::Mat src, Line line, cv::Rect area);
	cv::Mat drawLine(cv::Mat src, Line line);
	double getGradient();
	double getYintercept();
	double calcLineErr(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers);

//private variables
private:
	std::vector<double> xs;
	std::vector<double> ys;
	std::vector<bool> inliers;
	
	int nof_iter, err_thresh;

	double meanx;
	double meany;
	double std_deviationx;
	double std_deviationy;
	double correlation;
	cv::Mat points;

//private function
private:
	void calcMean(std::vector<double> points_x, std::vector<double> points_y,bool only_inliers);
	double calcGradient(bool only_inliers);
	double calcYintercept(bool only_inliers);
	void calcstdDeviation(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers);
	void calcCorrelation(std::vector<double> points_x, std::vector<double> points_y, bool only_inliers);
	double RANSAC(std::vector<double> points_x, std::vector<double> points_y);

};

class depthPlaneDetector{
	//public functions
	public : 
		//constructor
		depthPlaneDetector();
		depthPlaneDetector(cv::Mat* input, int w_ksize, int h_ksize);
		depthPlaneDetector(int w_ksize, int h_ksize);

		std::vector<Line> piecewiseLinearRegression(cv::Mat graphroi);
		double calcStandardDeviation(cv::Mat* input, cv::Size ksize, int start, int starty);
		cv::Size getWindowsize();
		cv::Mat	getReferenceImage();
		float getStandardDeviation();
		void setWindowsize(cv::Size ksize);
		std::vector<cv::Mat> searchDeviationDx(cv::Mat input, cv::Mat colorimg);
		cv::Mat displayDepthGraph(cv::Mat data, int startx,int starty);
		cv::Mat displayGraph(cv::Mat data);
				void setFoundKeyPt(bool found);
		bool isKeyPtFound();
		std::vector<cv::Rect> getExcerptWindow();
		std::vector<cv::Rect> searchDepthDeviation(cv::Mat depth_img, cv::Mat depth_graph);

	//private functions
	private : 
		void saveDepthFrame(const char* filename, cv::Mat* image, int framecount);
		std::vector<cv::Mat> findEdges(cv::Mat depthgraph, cv::Mat colorimg, double max=0.0f, std::vector<int> max_x=std::vector<int>());
		std::pair<int, int>	convertOffset(cv::Size src, cv::Size targ, int offsetx, int offsety);
		cv::Size depthPlaneDetector::convertSize(cv::Size src, cv::Size targ, int width, int height);
	//public variables
	public :
		int left_x,left_y,right_x, right_y;

	//private variables
	private : 
		std::vector<int> excerpt_xs;
		std::vector<cv::Rect> excerpts;
		bool foundKeyPt;
		float mean;
		float std_dev;
		cv::Size windowsize;
		cv::Mat* image;
};
