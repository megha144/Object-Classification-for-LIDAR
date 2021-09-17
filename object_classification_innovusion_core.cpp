/*

Feature Set for Velodyne HDL-64E [KITTI]
--------------------------------------------
1. Height
2. Width
3. Box_volume

Feature set for Innovusion LiDAR
--------------------------------------------
1. Height
2. Width
3. Box_volume
4. L1 - Scatterness of the object [eigen value based feature]
5. L2 - Linearness of the object  [eigen value based feature]
6. L3 - Surfaceness of the object [eigen value based feature]
7. Distance from ground - Height of the cluster centroid // derived from height in jupyter
8. The standard deviation of the distance from each point to the center of gravity of the object -- ?

Class Labels for Innovusion
----------------------------
Car, Pedestrian, Cyclist, Motorbike [unknown], Truck, 
Misc [all empty labels]

NOTE : Range from 10 Meter to 70 Meter for Feature Extraction


*/

#include "object_classification_innovusion_core.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

ObjectClassification::ObjectClassification(ros::NodeHandle &nh)
{
    //NEW INNOVUSION
    //subscribe tracked detected object array from tracking node
    sub_tracked_detected_obj_array_ = nh.subscribe("/detection/lidar_tracker/objects_with_class", 10, &ObjectClassification::extract_feature_from_array, this);

    ros::spin();
}

ObjectClassification::~ObjectClassification()
{
}

void ObjectClassification::Spin()
{
}

//LiDAR
//callback
void ObjectClassification::extract_feature_from_array(const autoware_msgs::DetectedObjectArray &detected_object_array)
{
    //create detected object array and detected object
    autoware_msgs::DetectedObjectArray detected_obj_array;
    autoware_msgs::DetectedObject detected_obj;

    //class_label to store groundtruth
    std::string class_label;
    //feature vector including all the features
    feature_vect features;
    //variables for each feature extraction
    double height, width, length, L1, L2, L3, ground_distance, eigen_curvature, eigen_entropy, 
    volume = 0, omnivariance, anisotropy, sum_eigen, norm_e1, norm_e2, norm_e3;
    //file_pointers for all the feature vector
    ofstream height_f1, width_f2, box_f3, label_f4, l1_f5, l2_f6, l3_f7, eigen_curve_f8, 
    eigen_entropy_f9,omnivariance_f10,anisotropy_f11,sum_eigen_f12;
    //txt file path - to store all the features
    std::string txt_file_path = "~//features_innovusion//";
    //calculate detected_object_array size
    int array_size;
    array_size = detected_object_array.objects.size();

    int frame_id = detected_object_array.frame_number;


    //asssign header and frame_number
    detected_obj.header = detected_object_array.header;
    detected_obj_array.header = detected_object_array.header;
    detected_obj_array.frame_number = detected_object_array.frame_number;

    //iterate through the detected_object_array
    for (size_t i = 0; i < array_size; i++)

    {

        detected_obj.pose_reliable = detected_object_array.objects[i].pose_reliable;
        detected_obj.pose.position.x = detected_object_array.objects[i].pose.position.x;

        //check for pose reliable to true for all the objects
        if (detected_obj.pose_reliable == true && (detected_obj.pose.position.x >= 10 &&
                                                   detected_obj.pose.position.x <= 70))
        {
            //get eigen_values,dimensions,groundtruth,centroid from array
            detected_obj.eigen_values = detected_object_array.objects[i].eigen_values;
            detected_obj.dimensions = detected_object_array.objects[i].dimensions;
            detected_obj.class_label_true = detected_object_array.objects[i].class_label_true;
            detected_obj.class_label_pred = detected_object_array.objects[i].class_label_pred;
            detected_obj.centroid_point.point.x = detected_object_array.objects[i].centroid_point.point.x;

            //class_label
            if (detected_obj.class_label_pred == "")
            {
                class_label = detected_obj.class_label_true;
            }
            else
            {
                class_label = detected_obj.class_label_pred;
            }
            //calculate features from array
            height = detected_obj.dimensions.x; 
            width = detected_obj.dimensions.y;  
            length = detected_obj.dimensions.z; 
            volume = height * width * length;
            L1 = detected_obj.eigen_values.x; // SORTED EIGEN VALUES
            L2 = detected_obj.eigen_values.x - detected_obj.eigen_values.y;
            L3 = detected_obj.eigen_values.y - detected_obj.eigen_values.z;
            sum_eigen = detected_obj.eigen_values.x + detected_obj.eigen_values.y + detected_obj.eigen_values.z;
            norm_e1 = detected_obj.eigen_values.x/sum_eigen;
            norm_e2 = detected_obj.eigen_values.y/sum_eigen;
            norm_e3 = detected_obj.eigen_values.z/sum_eigen;
            eigen_curvature = norm_e3 / (norm_e1 + norm_e2 + norm_e3);
            eigen_entropy =  - ((norm_e1 * log(norm_e1)) + (norm_e2 * log(norm_e2)) + (norm_e3 * log(norm_e3)));
            omnivariance = cbrt(norm_e1 * norm_e2 * norm_e3);
            anisotropy = (norm_e1 - norm_e3)/norm_e1;

            //extract features acording to groundtruth and store them to feature structure

            if (class_label != "Misc" && class_label != "" ) //TODO
            {   
                std::cout << "Extract features for class : " << class_label << " from frame : " << detected_obj_array.frame_number << "\n";
                features.height.push_back(height);
                features.width.push_back(width);
                features.class_name.push_back(class_label);
                features.box_volume.push_back(volume);
                features.L1.push_back(L1);
                features.L2.push_back(L2);
                features.L3.push_back(L3);
                features.eigen_curvature.push_back(eigen_curvature);
                features.eigen_entropy.push_back(eigen_entropy);
                features.omnivariance.push_back(omnivariance);
                features.anisotropy.push_back(anisotropy);
                features.sum_eigen.push_back(sum_eigen);

            }
            else
            {
                std::cout << "Do nothing"<< "\n";
            }
            
            //open txt file to store hight feature
            height_f1.open(txt_file_path + "height.txt", ios::in | ios::out | ios::app);
            for (double n : features.height)
            {
                if (height_f1.is_open())
                {
                    height_f1 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close height feature txt file
            height_f1.close();

            //open txt file to store width feature
            width_f2.open(txt_file_path + "width.txt", ios::in | ios::out | ios::app);
            for (double n : features.width)
            {
                if (width_f2.is_open())
                {
                    width_f2 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close width feature txt file
            width_f2.close();

            //open txt file to store volume feature
            box_f3.open(txt_file_path + "box_volume.txt", ios::in | ios::out | ios::app);
            for (double n : features.box_volume)
            {
                if (box_f3.is_open())
                {
                    box_f3 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close volume feature txt file
            box_f3.close();

            //open txt file to store class_label
            label_f4.open(txt_file_path + "class_label.txt", ios::in | ios::out | ios::app);
            for (std::string n : features.class_name)
            {
                if (label_f4.is_open())
                {
                    std::cout << "label_4 reads : " << n << "\n";

                    label_f4 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close class_label txt file
            label_f4.close();

            //open txt file to store l1 feature
            l1_f5.open(txt_file_path + "l1_feature.txt", ios::in | ios::out | ios::app);
            for (double n : features.L1)
            {
                if (l1_f5.is_open())
                {
                    std::cout << "l1_f5 reads : " << n << "\n";

                    l1_f5 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close l1 feature txt file
            l1_f5.close();

            //open txt file to store l2 feature
            l2_f6.open(txt_file_path + "l2_feature.txt", ios::in | ios::out | ios::app);
            for (double n : features.L2)
            {
                if (l2_f6.is_open())
                {
                    std::cout << "l2_f6 reads : " << n << "\n";

                    l2_f6 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close l2 feature txt file
            l2_f6.close();

            //open txt file to store l3 feature
            l3_f7.open(txt_file_path + "l3_feature.txt", ios::in | ios::out | ios::app);
            for (double n : features.L3)
            {
                if (l3_f7.is_open())
                {
                    std::cout << "l3_f7 reads : " << n << "\n";

                    l3_f7 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close l3 feature txt file
            l3_f7.close();
            //open txt file to store eigen_curvature feature
            eigen_curve_f8.open(txt_file_path + "eigen_curvature_feature.txt", ios::in | ios::out | ios::app);
            for (double n : features.eigen_curvature)
            {
                if (eigen_curve_f8.is_open())
                {
                    std::cout << "eigen_curve_f8 reads : " << n << "\n";

                    eigen_curve_f8 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close eigen_curve_f8 txt file
            eigen_curve_f8.close();

            //open txt file to store eigen_entropy feature
            eigen_entropy_f9.open(txt_file_path + "eigen_entropy.txt", ios::in | ios::out | ios::app);
            for (double n : features.eigen_entropy)
            {
                if (eigen_entropy_f9.is_open())
                {
                    std::cout << "eigen_entropy_f9 reads : " << n << "\n";

                    eigen_entropy_f9 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close eigen_entropy feature txt file
            eigen_entropy_f9.close();

            //open txt file to store omnivariance feature
            omnivariance_f10.open(txt_file_path + "omnivariance.txt", ios::in | ios::out | ios::app);
            for (double n : features.omnivariance)
            {
                if (omnivariance_f10.is_open())
                {
                    std::cout << "omnivariance_f10 reads : " << n << "\n";

                    omnivariance_f10 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close omnivariance feature txt file
            omnivariance_f10.close();

            //open txt file to store anisotropy feature
            anisotropy_f11.open(txt_file_path + "anisotropy.txt", ios::in | ios::out | ios::app);
            for (double n : features.anisotropy)
            {
                if (anisotropy_f11.is_open())
                {
                    std::cout << "anisotropy_f11 reads : " << n << "\n";

                    anisotropy_f11 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close anisotropy feature txt file
            anisotropy_f11.close();

             //open txt file to store sum_eigen feature
            sum_eigen_f12.open(txt_file_path + "sum_eigen.txt", ios::in | ios::out | ios::app);
            for (double n : features.sum_eigen)
            {
                if (sum_eigen_f12.is_open())
                {
                    std::cout << "sum_eigen_f12 reads : " << n << "\n";

                    sum_eigen_f12 << n << "\n";
                }
                else
                {
                    cout << "Unable to open file";
                }
            }
            //close sum_eigen feature txt file
            sum_eigen_f12.close();

        }
    }
}
