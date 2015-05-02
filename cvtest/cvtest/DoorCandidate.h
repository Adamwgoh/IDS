#include "stdafx.h"
#include "depthPlaneDetector.h"
#include "colorProcessor.h"
#include "Frame.h"

class DoorCandidate{
//public function
public:
	//constructor
	DoorCandidate();

	void storeLine(Line line_model);
	void storeDoorExtract(cv::Mat roi);
	void storeDoorDepth(double depth_val);
	void storeDoorColour(cv::Vec3b door_colour);
	
	std::vector<Line> getLineModels();
	cv::Mat getDoorExtract();
	double getDoorDepth();
	cv::Vec3b getDoorColour();

//private function
private:


//public variable
public:


//private variable
private:
	std::vector<Line> depth_lines;
	cv::Mat door;
	double door_depth;
	cv::Vec3b door_colour;

};