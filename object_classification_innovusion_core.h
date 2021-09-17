
#pragma once

#include <ros/ros.h> 
#include <pcl/PCLPointCloud2.h>       
#include <sensor_msgs/PointCloud2.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/common.h>
#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <pcl/filters/voxel_grid.h>
#include "jsk_recognition_msgs/BoundingBox.h"
#include "jsk_recognition_msgs/BoundingBoxArray.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <autoware_msgs/DetectedObject.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include "autoware_msgs/Centroids.h"
#include "autoware_msgs/CloudCluster.h"
#include "autoware_msgs/CloudClusterArray.h"
#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <std_msgs/Header.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include "math.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d.h>
#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <iostream>
#include <fstream>

using namespace autoware_msgs;
using namespace jsk_recognition_msgs;
using namespace sensor_msgs;



class ObjectClassification                      //define a class
{
  public:
    ObjectClassification(ros::NodeHandle& nh);
    ~ObjectClassification();                    // destructor
    void Spin();

  private:

    ros::Subscriber sub_tracked_detected_obj_array_; //NEW INNOVUSION

    //feature vector
    struct feature_vect {
    
    std::vector<std::string> class_name;
    std::vector<double> height;
    std::vector<double> width;
    std::vector<double> box_volume;
    std::vector<double> L1;
    std::vector<double> L2;
    std::vector<double> L3;
    std::vector<double> eigen_curvature;
    std::vector<double> eigen_entropy;
    std::vector<double> omnivariance;
    std::vector<double> anisotropy;
    std::vector<double> sum_eigen;
    };


    void extract_feature_from_array(const autoware_msgs::DetectedObjectArray &detected_object_array);//NEW INNOVUSION
  

};

