#include "planefitting.h"

void RANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr inputcloud){
	/**
	 * RANSAC Procedures
	 * 1) Create a Sample Consensus(SAC) model for detecting planes
	 * 2) Create a RANSAC algorithm, paramterized on epsilon = 3cm
	 * 3) computer best model
	 * 4) retrieve best set of inliers
	 * 5) retrieve correlated plane model coefficients
	 **/
	//create plane model pointer
	// TODO: input is a plane model, please fix.
	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model
	
		(new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (inputcloud));//input is the cloud

	//Create RANSAC object, insert plane model, 0.03 is 3 for epsilon in RANSAC parameter
	pcl::RandomSampleConsensus<pcl::PointXYZ> sac (model, 0.03);

	//perform segmentation
	bool result = sac.computeModel();

	//get inlier indices
	boost::shared_ptr<std::vector<int> > inliers (new std::vector<int>);
	sac.getInliers (*inliers);
	
	//get model coefficients
	Eigen::VectorXf coeff;
	sac.getModelCoefficients (coeff);




	//	/**
	//   * RANSAC
	//   **/
	//std::vector<int> inliers;
	////plane model sample consensus
	//pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
	//	pmodel (new pcl::SampleConsensusModelSphere<pcl::PointXYZ> (cloudptr));

	////RANSAC with the created plane model sample consensus
	//pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (pmodel);
	//ransac.setDistanceThreshold(0.1);
	//if(ransac.computeModel())	printf("model found\n");
	//ransac.getInliers(inliers);

	//pcl::PointCloud<pcl::PointXYZ>::Ptr pfinal (new pcl::PointCloud<pcl::PointXYZ>);
	////for(int i = 0; i < inliers.size(); i++){
	////	printf("inlier no %d : %d\n", i, inliers.at(i));
	////}
	////copies all inliers of model computer to another pointcloud
	//pcl::copyPointCloud<pcl::PointXYZ>(*cloudptr, inliers, *pfinal);
}