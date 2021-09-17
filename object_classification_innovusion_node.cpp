/** 
  * @megha
  */

#include "object_classification_innovusion_core.h"

//include other *.h files below according to different functions



int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_classification_innovusion"); //initialize ROS, specify the name of the node.
    ros::NodeHandle nh;                    //create a handle.
    ObjectClassification core(nh);

    return 0;
}




