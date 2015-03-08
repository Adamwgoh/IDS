#include <vector>
#include <exception>
#include <iostream>
#include <fstream>
#include <DepthSense.hxx>
#include <cv.h>
#include <highgui.h>
#include <vtkAlgorithm.h>
#include <vtkMapper.h>
#include <vtkDataSetAttributes.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkSmartPointerBase.h>
#include <vtkProp3D.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/exceptions.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>

#include "planefitting.h"

using namespace DepthSense;

class DeviceRetriever{
public:

DeviceRetriever();	
bool runNodes();
void getNodes(Context context, DepthNode d_node, ColorNode c_node);

//cv::
static void saveColorFrame(const char* filename, cv::Mat* image);
static void saveColorFrame(const char* filename, const char* url, cv::Mat* image);
static void saveDepthFrame(const char* filename, cv::Mat* image);
static void saveDepthFrame(const char* filename, const char* url, cv::Mat* image);
static void savePCDFile(const char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
static void savePCDFile(const char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
static void viewCloud(const char* cloudname, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
static void viewCloud(const char* cloudname, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);


//message sending functions here
bool isDeviceDisconnected();
bool isDeviceConnected();

private:
//event listeners for audio, color and depth
static void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data);
static void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data);

//functions to be called when node is connected or disconnected. When connected, call appriopriate configuration set
static void onNodeDisconnected(Device device, Device::NodeRemovedData data);
static void onNodeConnected(Device device, Device::NodeAddedData data);

static void onDeviceConnected(Context context, Context::DeviceAddedData data);
static void onDeviceDisconnected(Context context, Context::DeviceRemovedData data);

void configureDepthNode();
void configureColorNode();
void configureNode(Node node);

public:
	/**
	 * some format constants
	 **/
	//constant width for VGA format
	static int const VGA_WIDTH    = 640;
	//constant height for VGA format
	static int const VGA_HEIGHT   = 480;
	//constant width for QQVGA format
	static int const QQVGA_WIDTH  = 160;
	//constant height for QQVGA format
	static int const QQVGA_HEIGHT = 120;

	static Context d_Context;
	static ColorNode d_colorNode;
	static DepthNode d_depthNode;
	static bool d_isDeviceFound;
	uint32_t d_nofdepthframes;
	uint32_t d_nofcolorframes;
	int maxDepthThreshold;
	bool showFrame;

private:
};
