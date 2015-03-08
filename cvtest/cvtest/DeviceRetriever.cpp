#include "stdafx.h"
#include "DeviceRetriever.h"

	//constructors
	Context DeviceRetriever::d_Context = Context::create();
	ColorNode DeviceRetriever::d_colorNode = ColorNode();
	DepthNode DeviceRetriever::d_depthNode = DepthNode();
	bool DeviceRetriever::d_isDeviceFound = false;
	int dcount = 0;
	int ccount = 0;
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> pclviewer;
	void getRGBFrame(cv::Mat* src, cv::Mat& out, int frame_width, int frame_height);
	float packRGB(uint8_t* rgb);
	void cvtUVtoColorIdx(DepthSense::UV UV, int frame_height, int frame_width, int& uv_ColorIdx_x, int& uv_ColorIdx_y, int& uv_ColorIdx_Data );
	void cvtMapRes(FPVertex* src, FPVertex* output, int src_width, int src_height, int out_width, int out_height);
	void cvtMapRes(UV* src, UV* output, int src_width, int src_height, int out_width, int out_height);
	//local c++ variables
	pcl::PointCloud<pcl::PointXYZ> cloud;//create a new world coord points
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudptr(&cloud);
	uint8_t colorData[3*DeviceRetriever::VGA_WIDTH*DeviceRetriever::VGA_HEIGHT];
	UV uvMap_resize[DeviceRetriever::VGA_WIDTH*DeviceRetriever::VGA_HEIGHT];
	FPVertex vertPoints_resize[DeviceRetriever::VGA_WIDTH*DeviceRetriever::VGA_HEIGHT];
	
	DeviceRetriever::DeviceRetriever(){

	maxDepthThreshold = 2000,
	showFrame = true;
	
}

void DeviceRetriever::getNodes(Context context, DepthNode d_node, ColorNode c_node){
	printf("Getting nodes..\n");
	//context = Context::create("localhost");

	if(context.getDevices().size() >= 1){
		
		this->d_isDeviceFound = true;

	}
		//might not need to configure nodes as configured when nodes are connected
		vector<Node> nodes = context.getDevices().at(0).getNodes();
		printf("nodes : %d\n", context.getDevices().at(0).getNodes().size());
		for(int i = 0; i < nodes.size(); i++){
			configureNode(nodes.at(i));
		}
}

bool DeviceRetriever::runNodes(){
	
	if(!this->d_colorNode.isSet() && !this->d_depthNode.isSet()){
		getNodes(this->d_Context, this->d_depthNode, this->d_colorNode);
	}
	//this->d_Context = Context::create("localhost");

	//this->d_Context.deviceAddedEvent().connect(DeviceRetriever::onDeviceConnected);
	//this->d_Context.deviceRemovedEvent().connect(DeviceRetriever::onDeviceDisconnected);
	
	if(this->d_Context.getDevices().size() >= 1){
		this->d_isDeviceFound = true;

		this->d_Context.getDevices().at(0).nodeAddedEvent().connect(DeviceRetriever::onNodeConnected);
		this->d_Context.getDevices().at(0).nodeRemovedEvent().connect(DeviceRetriever::onNodeDisconnected);
	}
	this->d_Context.startNodes();
	this->d_Context.run();
	
	return true;
}


int rgbClamp(int val){
	int value = val;
	if(value > 255) value = 255;
	if(value < 0) value = 0;

	return value;
}

//event listeners for audio, color and depth
void DeviceRetriever::onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data){
	int32_t colorframe_width  = VGA_WIDTH;
	int32_t colorframe_height = VGA_HEIGHT;
	int32_t total_indices = colorframe_width*colorframe_height;

	cv::Mat* d_colorframe;
	cv::Mat* d_yuy2frame;
	//uint8_t colorData[3*VGA_WIDTH*VGA_HEIGHT];
	//I've made this variable static, assuming that the sample rate for both depth and color is the same
	//	if it isn't, think of an alternative

	//fills Mat with colorMap's data
	d_yuy2frame = new cv::Mat(cv::Size(colorframe_width, colorframe_height), CV_8UC3, (void*)(const uint8_t*)data.colorMap);
	getRGBFrame(d_yuy2frame, *d_colorframe, colorframe_width, colorframe_height);
	std::ostringstream ss;
	ss << "rawdata\\";
	ss << ccount;
	ss << "cframe.jpg";
	cv::String url = ss.str();
	cv::imwrite(url,*d_colorframe);
	ccount++;
	//converts raw data from color frame into RGB Mat
	
	

	//
	//for(int index = 0; index < colorframe_width*colorframe_height; index++){
	//	colorData[3*index]	 = data.colorMap[3*index + 2];//R
	//	colorData[3*index+1] = data.colorMap[3*index + 1];//G
	//	colorData[3*index+2] = data.colorMap[3*index];//B
	//}


}


void DeviceRetriever::onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data){
	
	int32_t depthframe_width  = QQVGA_WIDTH;
	int32_t depthframe_height = QQVGA_HEIGHT;
	ofstream depthRawvals;
	std::ostringstream url;
	url << "rawdata\\";
	url << dcount;
	url << "depthrawvals.txt";
	cv::String urls = url.str();
	depthRawvals.open(urls);
	std::ostringstream ss;
	printf("depth sample\n");
	cv::Mat depthframe = *new cv::Mat(cv::Size(depthframe_width, depthframe_height), CV_8UC1);
	for(int index = 0; index < depthframe_width*depthframe_height; index++){
		
		//printf("depth sample%d\n", index);
		//gets data into a text file
		if(depthRawvals.is_open()){

			ss << "dataPoint ";
			ss << index;
			ss << " of ";
		    ss << dcount ;
			ss << " depth sample	";
			ss << data.depthMap[index];
			ss << "\n";
		}
		if(data.depthMap[index] != 32001){
			depthframe.data[index]	 = data.depthMap[index];
		}else{
			depthframe.data[index] = data.depthMap[index];
		}
		
	}
	ss << "=====================================\n\n";
	depthRawvals << ss.str();
	depthRawvals.close();
	dcount++;
	printf("done\n");
	cv::waitKey(30);
	cv::imshow("test", depthframe);
	cv::waitKey(0);
	ostringstream stream;
	stream << "rawdata\\";
	stream << dcount;
	stream << "depthframe.jpg";
	cv::String dframe = stream.str();
	cv::imwrite(dframe, depthframe);

	//cloud.resize(data.verticesFloatingPoint.size());
	//cloud.points.resize(data.vertices.size());
	//// TODO: is there any exceptions to catch here?
	//for(int i = 0; i < data.verticesFloatingPoint.size(); i++){
	//	
	//			cloud.points[i].x = (float) data.verticesFloatingPoint[i].x;
	//			cloud.points[i].y = (float) data.verticesFloatingPoint[i].y;
	//			cloud.points[i].z = (float) data.verticesFloatingPoint[i].z;
	//			//printf("x : %f, y : %f, z : %f \n", cloud.points[i].x,cloud.points[i].y, cloud.points[i].z);
	//}	
	//UV* uv_map;
	//FPVertex* vertFlt_points;

	//get UV values
	//for(int i = 0; i < data.uvMap.size(); i++){
	//	uv_map[i] = data.uvMap[i];
	//}

	//get raw vertices points
	//for(int i = 0; i < data.verticesFloatingPoint.size(); i++){
	//	vertFlt_points[i].x = data.verticesFloatingPoint[i].x;
	//	vertFlt_points[i].y = data.verticesFloatingPoint[i].y;
	//	vertFlt_points[i].z = data.verticesFloatingPoint[i].z;
	//}

	//cvtMapRes(vertFlt_points, vertPoints_resize, QQVGA_WIDTH, QQVGA_HEIGHT, VGA_WIDTH, VGA_HEIGHT);
	//cvtMapRes(uv_map, uvMap_resize, QQVGA_WIDTH, QQVGA_HEIGHT,VGA_WIDTH,VGA_HEIGHT);

	//pcl::PointCloud<pcl::PointXYZRGB> testcloud;
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr testcloud_ptr(&testcloud);
	//testcloud.is_dense = false;//clouds have no NaN or Inf values. Important because RANSAC cannot have NaN values
	//testcloud.width = VGA_WIDTH;
	//testcloud.height = VGA_HEIGHT;
	//testcloud.resize(data.verticesFloatingPoint.size());

	//get color values
	//int cx, cy, cInd;
	//get vertices, and color and pack into the cloud
	//for(int i = 0; i < VGA_WIDTH*VGA_HEIGHT; i++){

	//	cvtUVtoColorIdx(uvMap_resize[i], VGA_WIDTH, VGA_HEIGHT, cx,cy,cInd);
	//	testcloud.points[i].x = vertPoints_resize[i].x;
	//	testcloud.points[i].y = vertPoints_resize[i].y;
	//	testcloud.points[i].z = vertPoints_resize[i].z;

	//	testcloud.points[i].rgb = packRGB(&colorData[3*i]);//multiply 3 because data will be packed?
	//}

	//viewCloud("testcloud", testcloud_ptr);

}

//functions to be called when node is connected or disconnected. When connected, call appriopriate configuration set
void DeviceRetriever::onNodeDisconnected(Device device, Device::NodeRemovedData data){
	if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == d_colorNode))
		d_colorNode.unset();
	if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == d_depthNode))
		d_depthNode.unset();
    printf("node disconnected\n");
}

void DeviceRetriever::onNodeConnected(Device device, Device::NodeAddedData data){

	//configurenode(data.node);
}

void DeviceRetriever::onDeviceConnected(Context context, Context::DeviceAddedData data){
	//TODO : when this is called, call a new function that sends acknowledgement
	if(!d_isDeviceFound){
		//data.device.nodeAddedEvent().connect(DeviceRetriever::onNodeConnected);
		//data.device.nodeRemovedEvent().connect(DeviceRetriever::onNodeDisconnected);

		d_isDeviceFound = true;
		printf("device connected\n");
	}
}

void DeviceRetriever::onDeviceDisconnected(Context context, Context::DeviceRemovedData data){
	//TODO : when this is called, call a new function that sends acknowledgement
	if(d_isDeviceFound){
		d_isDeviceFound = false;
		printf("device disconnected\n");
	}
}

void DeviceRetriever::configureDepthNode(){

	FrameFormat depthFormat = FRAME_FORMAT_QQVGA;
	this->d_depthNode.newSampleReceivedEvent().connect(DeviceRetriever::onNewDepthSample);
	this->d_depthNode.setEnableDepthMap(true);
	this->d_depthNode.setEnableUvMap(true);
	this->d_depthNode.setEnableVertices(true);
	this->d_depthNode.setEnableVerticesFloatingPoint(true);
	this->d_depthNode.setEnableUvMap(true);

	//cloud.points.resize((160*120));
	cloud.width = 160;
	cloud.height = 120;
	cloud.is_dense = false;
	//get depth node config and set it
	DepthNode::Configuration config = d_depthNode.getConfiguration();
	config.frameFormat = depthFormat;
	config.framerate = 60;
	config.mode = DepthNode::CAMERA_MODE_CLOSE_MODE;

	try{
		this->d_Context.requestControl(d_depthNode,0);
		this->d_depthNode.setConfiguration(config);
	}
	catch(ConfigurationException e)			{printf("ConfigurationException. Error message is %s\n", e.what());}
	catch(IOException e)					{printf("IOException. Error message is %s\n", e.what());}
	catch(TimeoutException e)				{printf("TimeoutException. Error message is %s\n", e.what()); }
	catch(UnauthorizedAccessException e)	{printf("UnauthorizedAccess. Error message is %s\n", e.what()); }
	catch(StreamingException e)				{printf("StreamingException. Error message is %s\n", e.what()); }
}


void DeviceRetriever::configureColorNode(){
	FrameFormat colorFormat = FRAME_FORMAT_VGA;
	this->d_colorNode.newSampleReceivedEvent().connect(DeviceRetriever::onNewColorSample);
	this->d_colorNode.setEnableColorMap(true);
	this->d_colorNode.setEnableCompressedData(true);

	//get color node config and set it
	ColorNode::Configuration config =  this->d_colorNode.getConfiguration();
	config.compression = COMPRESSION_TYPE_YUY2;
	config.frameFormat = colorFormat;
	config.framerate = 30;
	config.powerLineFrequency = POWER_LINE_FREQUENCY_50HZ;

	try{
		this->d_Context.requestControl(this->d_colorNode,0);
		//this->d_colorNode.setConfiguration(config);
		this->d_colorNode.setBrightness(0);
		this->d_colorNode.setContrast(5);
		this->d_colorNode.setSaturation(5);
		this->d_colorNode.setHue(0);
		this->d_colorNode.setConfiguration(config);
	}
	catch(ArgumentException e)				{printf("argument exception. Error message is %s\n", e.what());}
	catch (std::bad_alloc& e)				{printf("bad alloc. Error message is %s\n", e.what());}
	catch(ConfigurationException e)			{printf("ConfigurationException. Error message is %s\n", e.what());}
	catch(IOException e)					{printf("IOException. Error message is %s\n", e.what());}
	catch(TimeoutException e)				{printf("TimeoutException. Error message is %s\n", e.what()); }
	catch(UnauthorizedAccessException e)	{printf("UnauthorizedAccess. Error message is %s\n", e.what()); }
	catch(StreamingException e)				{printf("StreamingException. Error message is %s\n", e.what()); }
}

void DeviceRetriever::configureNode(Node node){
	
	if (node.is<ColorNode>() && (!this->d_colorNode.isSet()) ){

		this->d_colorNode = node.as<ColorNode>();
		configureColorNode();
		this->d_Context.registerNode(this->d_colorNode);
		printf("registered color Node\n");
	}

	if (node.is<DepthNode>() && (!this->d_depthNode.isSet()) ){

		this->d_depthNode = node.as<DepthNode>();
		configureDepthNode();
		this->d_Context.registerNode(this->d_depthNode);
		printf("registered depth node\n");
	}
}

//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension. Also add a slash after url!
void DeviceRetriever::saveColorFrame(const char* filename, const char* url, cv::Mat* image){
	std::string temp(url);
	temp.append(filename);
	temp.append(".jpg");
	cv::imwrite( temp, *image );
}

//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension. Also add a slash after url!
void DeviceRetriever::saveDepthFrame(const char* filename, const char* url, cv::Mat* image){
	std::string temp(url);
	temp.append(filename);
	temp.append(".jpg");
	cv::imwrite( temp, *image );
}

//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension
void DeviceRetriever::saveColorFrame(const char* filename, cv::Mat* image){
	std::string temp(filename);
	temp.append(".jpg");
	cv::imwrite( temp, *image );
}


//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension
void DeviceRetriever::saveDepthFrame(const char* filename, cv::Mat* image){
	cv::String url(filename);
	url.append(filename,".jpg");
	cv::imwrite( url, *image );
}

//saves frame. Note that extension of .jpg is added automatically. Do not add anymore .jpg extension.
void DeviceRetriever::savePCDFile(const char* filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
		try{
			cv::String file = file.append(filename,".pcd");
			pcl::io::savePCDFileBinary(file, *cloud);
		}catch(pcl::IOException e){
			printf("IOException. Message : %s", e.what());
		}
}


void DeviceRetriever::savePCDFile(const char* filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
		try{
			cv::String file = file.append(filename,".pcd");
			pcl::io::savePCDFileBinary(file, *cloud);
		}catch(pcl::IOException e){
			printf("IOException. Message : %s", e.what());
		}
}

void DeviceRetriever::viewCloud(const char* cloudname, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
	pcl::visualization::CloudViewer viewer(cloudname);
	viewer.showCloud(cloud, cloudname);

	while(!viewer.wasStopped()){
		
	}
}

void DeviceRetriever::viewCloud(const char* cloudname, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){
	pcl::visualization::CloudViewer viewer(cloudname);
	viewer.showCloud(cloud, cloudname);

	while(!viewer.wasStopped()){
		
	}
}



/**
 *	gets the U and V coordinates from the depth sample, converts it and retrieves the x and y coordinate of the pixel's color coordinates.
 *  uv_ColorIdx_x represents the x coordinate of the converted U from given UvMap.
 *  uv_COlorIdx_y represents the y coordinate of the converted V from given UvMap.
 *  uv_ColorIdx_Data represents the data index of the ColorFrame data, given the x and y coords of the converted UV coords.
 **/ 
void cvtUVtoColorIdx(DepthSense::UV UV, int frame_height, int frame_width, int& uv_ColorIdx_x, int& uv_ColorIdx_y, int& uv_ColorIdx_Data ){
	DepthSense::UV uvMap = UV;

	uv_ColorIdx_x = (int) (uvMap.u * (float) frame_width);
	uv_ColorIdx_y = (int) (uvMap.v * (float) frame_height);
	uv_ColorIdx_Data = (uv_ColorIdx_x)*frame_width + uv_ColorIdx_y;

}

// TODO: remember to credit properly. URL : ph4m/DepthSenseGrabber
//converts Matrix/map resolution
void cvtMapRes(UV* src, UV* output, int src_width, int src_height, int out_width, int out_height){
	//get the ratio between two resolution
	float stepWidth=(float)(src_width-1)/(float)(out_width-1);
    float stepHeight=(float)(src_height-1)/(float)(out_height-1);

    for (int x=0;x<out_width;x++)
    {
		// X scaled down by the ratio
        float fx=x*stepWidth;
		//if converting the scaled-x, is there spill over? if yes, use a ceiling value
        float dx=fx-(int)fx;
        int ffx = floor(fx);
        int cfx = ceil(fx);
        for (int y=0;y<out_height;y++)
        {
			//same thing as above, scaled by ratio, check for spills
            float fy=y*stepHeight;
            float dy=fy-(int)fy;

			//take ceiling and floor value of any float values for later evaluation
            int ffy = floor(fy);
            int cfy = ceil(fy);

            float u1,u2,u3,u4;
			float v1,v2,v3,v4;
			//4 possible combinations
            u1 = src[ffx + ffy*src_width].u;
            u2 = src[cfx + ffy*src_width].u;
            u3 = src[ffx + cfy*src_width].u;
            u4 = src[cfx + cfy*src_width].u;
			v1 = src[ffx + ffy*src_width].v;
            v2 = src[cfx + ffy*src_width].v;
            v3 = src[ffx + cfy*src_width].v;
            v4 = src[cfx + cfy*src_width].v;

			//if there's spill in scaled-x, take the ceiling val of scaled-x, otherwise take floor val of sacled-x
            float UvalT1 = dx*u2 + (1-dx)*u1;
            float UvalT2 = dx*u4 + (1-dx)*u3;

			float VvalT1 = dx*v2 + (1-dx)*v1;
			float VvalT2 = dx*v4 + (1-dx)*v3;

			//same procedure but for scaled-y
			float u = dy*UvalT2 + (1-dy)*UvalT1;
			float v = dy*VvalT2 + (1-dy)*VvalT1;

            output[x + y*out_height] = UV(u,v);
        }
    }
}

//converts Matrix/map resolution
void cvtMapRes(FPVertex* src, FPVertex* output, int src_width, int src_height, int out_width, int out_height){
	//get the ratio between two resolution
	float stepWidth=(float)(src_width-1)/(float)(out_width-1);
    float stepHeight=(float)(src_height-1)/(float)(out_height-1);

    for (int x=0;x<out_width;x++)
    {
		// X scaled down by the ratio
        float fx=x*stepWidth;
		//if converting the scaled-x, is there spill over? if yes, use a ceiling value
        float dx=fx-(int)fx;
        int ffx = floor(fx);
        int cfx = ceil(fx);
        for (int y=0;y<out_height;y++)
        {
			//same thing as above, scaled by ratio, check for spills
            float fy=y*stepHeight;
            float dy=fy-(int)fy;

			//take ceiling and floor value of any float values for later evaluation
            int ffy = floor(fy);
            int cfy = ceil(fy);

            float x1,x2,x3,x4;
			float y1,y2,y3,y4;
			float z1,z2,z3,z4;
			//4 possible combinations for all x, y and z
            x1 = src[ffx + ffy*src_width].x;
            x2 = src[cfx + ffy*src_width].x;
			x3 = src[ffx + cfy*src_width].x;
			x4 = src[cfx + cfy*src_width].x;

			y1 = src[ffx + ffy*src_width].y;
			y2 = src[cfx + ffy*src_width].y;
			y3 = src[ffx + cfy*src_width].y;
            y4 = src[cfx + cfy*src_width].y;
			
			z1 = src[ffx + ffy*src_width].z;
			z2 = src[cfx + ffy*src_width].z;
            z3 = src[ffx + cfy*src_width].z;
			z4 = src[cfx + cfy*src_width].z;

			//if there's spill in scaled-x, take the ceiling val of scaled-x, otherwise take floor val of sacled-x
            float xValT1 = dx*x2 + (1-dx)*x1;
            float xValT2 = dx*x4 + (1-dx)*x3;

			float yValT1 = dx*y2 + (1-dx)*y1;
			float yValT2 = dx*y4 + (1-dx)*y3;

			float zValT1 = dx*z2 + (1-dx)*z1;
			float zValT2 = dx*z4 + (1-dx)*z3;

			//same procedure but for scaled-y
			float zfx = dy*xValT2 + (1-dy)*xValT1;
			float zfy = dy*yValT2 + (1-dy)*yValT1;
			float zfz = dy*zValT2 + (1-dy)*zValT1;


            output[x + y*out_height].x = zfx;
			output[x + y*out_height].y = zfy;
			output[x + y*out_height].z = zfz;

        }
    }
}

void getRGBFrame(cv::Mat* src, cv::Mat& out, int frame_width, int frame_height){

	//YUY2 uses 4:2:2 sampling standard, that is : Y takes 4bits, U and V takes 2 bits, total 16bytes
	//2pixels represented in 1 micropixels
	//sequence is Y0U0Y1V0 Y2U1Y3V1

	/**
	 * Extract YUV components
		Y0 U0 Y1 V0		Y2 U2 Y3 V2
	**/
	int datasize = frame_width*frame_height*3*sizeof(char);
	//Mat* d_yuy2frame = new Mat(Size(frame_width, frame_height), CV_8UC3, 
	int y0,y1,u,v;
	int r1,g1,b1;
	int r2,g2,b2;

	int step = 0;
	unsigned char *colorframe = (unsigned char*) malloc(datasize);
	//Fill empty Mat with 0s. Similar to Zeros();
	for(int i=0; i<640*480*3*sizeof(char); i++) colorframe[i] = 0;

		for(int i = 0; i < (frame_height/2); i++){
			for(int j = 0; j < frame_width; j++){

				int steps = ((i*frame_width) + j) * 4;	
			
				y0 = src->data[steps] & 0xff;	// 0xff is a bitmask to limit value to range 0~255
				v  = src->data[steps+1] & 0xff;
				y1 = src->data[steps+2] & 0xff;
				u  = src->data[steps+3] & 0xff;
			
				//printf("y0 : %d, y1 : %d, u : %d, v : %d\n", y0,y1,u,v);
				//printf("one elemSize : %d\n", d_yuy2frame.elemSize());

			
				int r1 = rgbClamp((int) (1.164*(y0-16) + 1.567*(v-128)));
				int g1 = rgbClamp((int) (1.164*(y0-16) - 0.798 * (v-128) - 0.384*(u-128)));
				int b1 = rgbClamp((int) (1.164*(y0-16) + 1.980 * (u-128)));
				int r2 = rgbClamp((int) (1.164*(y1-16) + 1.567*(v-128)));
				int g2 = rgbClamp((int) (1.164*(y1-16) - 0.798 * (v-128) - 0.384*(u-128)));
				int b2 = rgbClamp((int) (1.164*(y1-16) + 1.980 * (u-128)));
				//printf("r : %d, g : %d, b : %d\n", r1,g1,b1);

				colorframe[step++] = r1;
				colorframe[step++] = g1;
				colorframe[step++] = b1;
				colorframe[step++] = r1;
				colorframe[step++] = g1;
				colorframe[step++] = b1;
			}
		}
		
				out = *new cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3);
				out.data = colorframe;
		
	}

float packRGB(uint8_t* rgb){
	uint32_t rgb32 = ((uint32_t)rgb[0] << 16 | (uint32_t) rgb[1] << 8 | (uint32_t)rgb[2]);
	return *reinterpret_cast<float*>(&rgb32);
}

int main(int args, char* argc[]){

	DeviceRetriever* retrieve = new DeviceRetriever();

	retrieve->runNodes();

	return 0;
}