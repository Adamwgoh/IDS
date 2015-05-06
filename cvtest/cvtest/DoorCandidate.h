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
	void storeDoorColour(cv::Vec3b door_colour);
	void storeDoorGraph(cv::Mat graph);

	std::vector<Line> getLineModels();
	cv::Mat getDoorExtract();
	cv::Mat getDoorGraph();
	cv::Vec3b getDoorColour();
	
	void setHasDoor(bool hasdoor);
	bool hasDoor();
	void printLines();
//private function
private:


//public variable
public:


//private variable
private:
	bool doorfound;
	std::vector<Line> depth_lines;
	cv::Mat door;
	cv::Mat door_graph;
	cv::Vec3b door_colour;

};