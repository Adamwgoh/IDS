#include "stdafx.h"
#include "Frame.h"
#include "depthPlaneDetector.h"
#include "ColorProcessor.h"
#include "DoorCandidate.h"

class DoorDetector{
//public functions
public:
	DoorDetector();
	DoorCandidate hasDoor(cv::Mat colour_src, cv::Mat depth_src);
	std::vector<Line> findDepthLines(cv::Mat depth_src, cv::Mat depth_graph);
	std::vector<cv::Rect> getSegments(cv::Mat depth_graph, std::vector<cv::Rect> rois);
	std::vector<std::pair<cv::Rect, Line>> getExcerpts(cv::Mat depthgraph, std::vector<cv::Rect> segments, std::vector<Line> models);
	const char* DetectTexts(cv::Mat colorpano);

//private functions
private:
	double findDepth(cv::Mat depthroi);
	std::vector<cv::Vec3b> findColours();
	
//public variables
public:
	ColorProcessor cProc;
	depthPlaneDetector dDetector;

//private functions
private:
	std::vector<DoorCandidate> candidates;
};