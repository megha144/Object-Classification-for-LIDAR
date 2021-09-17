#!/usr/bin/env python

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, accuracy_score
import cPickle
import numpy as np
import rospy
from std_msgs.msg import String
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from autoware_msgs.msg import CloudClusterArray, CloudCluster
from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters
import pykitti
from sklearn import preprocessing
import itertools
import pandas as pd
from datetime import datetime
import math


"""

# Innovusion Features
-----------------------------------------------
Simple features:
---------------------

1. Height of the cluster
2. Width of the cluster
3. Box_volume of the cluster

Eigen features:
-----------------------

1. L1
2. L2
3. L3
4. Change of Curvature
5. Eigen Entropy
6. Omnivariance
7. Anisotropy
8. Sum of Eigen Values

All features:
-------------------------

1. Height of the cluster
2. Width of the cluster
3. Box_volume of the cluster
4. L1
5. L2
6. L3
7. Change of Curvature
8. Eigen Entropy
9. Omnivariance
10. Anisotropy
11. Sum of Eigen Values


# Data Distribution - Innovusion
------------------------------------------------

1. Unbalanced - without feature scaling
2. Unbalanced - with feature scaling
3. Balanced - without feature scaling
4. Balanced - with feature scaling

# Class Encoding
--------------------------
#Car : 0, Cyclist : 1, Misc : 2, Motorbike : 3, Pedestrian : 4 



Note: Currently this code uses SVM with feature scaling (balanced data) - 4_SVM_balanced_with_scaling_all_features.pkl ,



"""


# publish detected object array after prediction
pub = rospy.Publisher("/detection/lidar_tracker/objects_with_class_megha",
                      DetectedObjectArray, queue_size=10)


#global variables
predicted_class = ""
expected = []
predicted = []

# trained model path [1. CHANGE PATH]
model_base_path = '~/online_class_prediction/src/trained_models/trained_models_new_innovusion'

# balanced or unbalanced data
model_path_balanced = model_base_path + '/balanced'
model_path_unbalanced = model_base_path + '/unbalanced'

# path with feature scaled and without feature scaled [2. CHANGE FOR BALANCED/UNBALANCED DATA]
model_path_with_scaling = model_path_balanced + '/normalized_1'
model_path_without_scaling = model_path_balanced + '/unnormalized'

# feature scaling [3. CHANGE FOR FEATURE SCALING/ DEFAULT - TRUE]
feature_scaling = True

# select feature set [4. CHANGE FOR FEATURE SET]
simple_feature = True
eigen_feature = False
all_feature = False

if(simple_feature):
    SVM_Path = model_path_with_scaling+'/svm/simple_features/'
    # selected trained model
    selected_model = SVM_Path+'4_SVM_balanced_with_scaling_simple_features.pkl'
    # selected trained scaler per each feature
    selected_scaler_h = SVM_Path+'4_SVM_scaler_h_simple.sav'
    selected_scaler_w = SVM_Path+'4_SVM_scaler_w_simple.sav'
    selected_scaler_v = SVM_Path+'4_SVM_scaler_v_simple.sav'

elif(eigen_feature):
    SVM_Path = model_path_with_scaling+'/svm/eigen_features/'
    # selected trained model
    selected_model = SVM_Path+'4_SVM_balanced_with_scaling_eigen_features.pkl'
    # selected trained scaler per each feature
    selected_scaler_l1 = SVM_Path+'4_SVM_scaler_l1_eigen.sav'
    selected_scaler_l2 = SVM_Path+'4_SVM_scaler_l2_eigen.sav'
    selected_scaler_l3 = SVM_Path+'4_SVM_scaler_l3_eigen.sav'
    selected_scaler_eigen_curve = SVM_Path+'4_SVM_scaler_eigen_curve_eigen.sav'
    selected_scaler_eigen_entropy = SVM_Path+'4_SVM_scaler_eigen_entropy_eigen.sav'
    selected_scaler_omnivari = SVM_Path+'4_SVM_scaler_omnivari_eigen.sav'
    selected_scaler_anisotr = SVM_Path+'4_SVM_scaler_anisotr_eigen.sav'
    selected_scaler_sum_eigen = SVM_Path+'4_SVM_scaler_sum_eigen_eigen.sav'
else:
    SVM_Path = model_path_with_scaling+'/svm/all_features/'
    # selected trained model
    selected_model = SVM_Path+'4_SVM_balanced_with_scaling_all_features.pkl'
    # selected trained scaler per each feature
    selected_scaler_h = SVM_Path+'4_SVM_scaler_h_all.sav'
    selected_scaler_w = SVM_Path+'4_SVM_scaler_w_all.sav'
    selected_scaler_v = SVM_Path+'4_SVM_scaler_v_all.sav'

    selected_scaler_l1 = SVM_Path+'4_SVM_scaler_l1_all.sav'
    selected_scaler_l2 = SVM_Path+'4_SVM_scaler_l2_all.sav'
    selected_scaler_l3 = SVM_Path+'4_SVM_scaler_l3_all.sav'
    selected_scaler_eigen_curve = SVM_Path+'4_SVM_scaler_eigen_curve_all.sav'
    selected_scaler_eigen_entropy = SVM_Path+'4_SVM_scaler_eigen_entropy_all.sav'
    selected_scaler_omnivari = SVM_Path+'4_SVM_scaler_omnivari_all.sav'
    selected_scaler_anisotr = SVM_Path+'4_SVM_scaler_anisotr_all.sav'
    selected_scaler_sum_eigen = SVM_Path+'4_SVM_scaler_sum_eigen_all.sav'


"""
@param = prediction_class (integer encoded predicted class provided by trained model) 
This function performs reverse mapping of integer encoding to string

"""


def reverse_mapping(prediction_classes):

    if(prediction_classes == 0):
        predicted_class = 'Car'
    elif(prediction_classes == 1):
        predicted_class = 'Cyclist'
    elif(prediction_classes == 3):
        predicted_class = 'Motorbike'
    elif(prediction_classes == 4):
        predicted_class = 'Pedestrian'
    else:
        predicted_class = 'None'

    return predicted_class


"""
@param = detected_object_array (detetected object array published by tracking node) 
subscriber_topic : /detection/lidar_tracker/objects
Publisher_topic : /detection/lidar_tracker/objects_with_class_megha
This callback function predicts class_labels from trained_model and pushed predicted_labels back to detected_object_array

"""


def callback(detected_object_array):

    print("Python Online Class Prediction node")

    # store detected_object_array length
    detected_array = detected_object_array
    detected_array.header = detected_object_array.header
    array_length = len(detected_object_array.objects)

    #detected_array.frame_number = detected_object_array.frame_number

    # loop through each detetected object inside detected_object_array
    for index in range(array_length):

        pose_reliable = detected_object_array.objects[index].pose_reliable
        # checked for tracked bounding boxes
        # if detected_object_array.objects[index].pose.position.x >=10 and detected_object_array.objects[index].pose.position.x <=70:
        if pose_reliable == True:

            # store bounding box dimensions
            detected_array.objects[index].class_label_pred = detected_object_array.objects[index].class_label_pred

            detected_array.objects[index].score = detected_object_array.objects[index].score

            detected_object_dimensions = detected_object_array.objects[index].dimensions

            # extract incoming feature from tracking
            e1 = detected_object_array.objects[index].eigen_values.x
            e2 = detected_object_array.objects[index].eigen_values.y
            e3 = detected_object_array.objects[index].eigen_values.z
            height = detected_object_dimensions.x
            width = detected_object_dimensions.y
            length = detected_object_dimensions.z
            volume = height * width * length

            if (e1 and e2 and e3 != 0):
                L1 = e1
                L2 = e1 - e2
                L3 = e2 - e3
                sum_eigen = e1 + e2 + e3
                norm_e1 = e1/sum_eigen
                norm_e2 = e2/sum_eigen
                norm_e3 = e3/sum_eigen
                eigen_curvature = norm_e3 / (norm_e1 + norm_e2 + norm_e3)
                #eigen_entropy = - ((norm_e1 * math.log(norm_e1)) +
                #            (norm_e2 * math.log(norm_e2)) + (norm_e3 * math.log(norm_e3)))

                #omnivariance = (norm_e1 * norm_e2 * norm_e3)**(1./3.)
                #anisotropy = (norm_e1 - norm_e3)/norm_e1
                # load the model from disk
                with open(selected_model, 'rb') as file:
                    # pickled model from jupyter
                    pickle_model = cPickle.load(file)

                    if pickle_model:
                        # feature scaling
                        if(feature_scaling == True):
                            if(simple_feature == True):
                                # Load the scaler for all the features

                                scaler_h = cPickle.load(
                                    open(selected_scaler_h, 'rb'))
                                scaler_w = cPickle.load(
                                    open(selected_scaler_w, 'rb'))
                                scaler_v = cPickle.load(
                                    open(selected_scaler_v, 'rb'))
                                # transform incoming features according to scaling
                                scaled_h = scaler_h.transform(
                                    np.array(height).reshape(1, 1))
                                scaled_w = scaler_w.transform(
                                    np.array(width).reshape(1, 1))
                                scaled_v = scaler_v.transform(
                                    np.array(volume).reshape(1, 1))

                                # create feature array
                                simple_feature_array = np.array(
                                    [scaled_h, scaled_w, scaled_v]).reshape(1, 3)
                                # prediction
                                prediction = pickle_model.predict(
                                    simple_feature_array)

                                # show predicted outputs

                                for i in range(len(simple_feature_array)):

                                    # reverse mapping of class labels from integer to string
                                    predicted_class = reverse_mapping(
                                        prediction[i])
                                    print("Prediction_class : {} | predicted_class : {} | frame : {}".format(
                                        prediction[i], predicted_class, detected_array.frame_number))
                                    # push predicted class to class_label_pred
                                    detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)

                                if(eigen_feature == True):
                                    scaler_l1 = cPickle.load(
                                        open(selected_scaler_l1, 'rb'))
                                    scaler_l2 = cPickle.load(
                                        open(selected_scaler_l2, 'rb'))
                                    scaler_l3 = cPickle.load(
                                        open(selected_scaler_l3, 'rb'))
                                    scaler_eigen_curve = cPickle.load(
                                        open(selected_scaler_eigen_curve, 'rb'))
                                    scaler_eigen_entropy = cPickle.load(
                                        open(selected_scaler_eigen_entropy, 'rb'))
                                    scaler_omnivariance = cPickle.load(
                                        open(selected_scaler_omnivari, 'rb'))
                                    scaler_anisotropy = cPickle.load(
                                        open(selected_scaler_anisotr, 'rb'))
                                    scaler_sum_eigen = cPickle.load(
                                        open(selected_scaler_sum_eigen, 'rb'))
                                    # transform incoming features according to scaling
                                    scaled_l1 = scaler_l1.transform(
                                        np.array(L1).reshape(1, 1))
                                    scaled_l2 = scaler_l2.transform(
                                        np.array(L2).reshape(1, 1))
                                    scaled_l3 = scaler_l3.transform(
                                        np.array(L3).reshape(1, 1))
                                    scaled_eigen_curve = scaler_eigen_curve.transform(
                                        np.array(eigen_curvature).reshape(1, 1))
                                    scaled_eigen_entropy = scaler_eigen_entropy.transform(
                                        np.array(eigen_entropy).reshape(1, 1))
                                    scaled_omnivariance = scaler_omnivariance.transform(
                                        np.array(omnivariance).reshape(1, 1))
                                    scaled_anisotropy = scaler_anisotropy.transform(
                                        np.array(anisotropy).reshape(1, 1))
                                    scaled_sum_eigen = scaler_sum_eigen.transform(
                                        np.array(sum_eigen).reshape(1, 1))

                                    # create feature array
                                    eigen_feature_array = np.array(
                                        [scaled_l1, scaled_l2, scaled_l3, scaled_eigen_curve, scaled_eigen_entropy, scaled_omnivariance, scaled_anisotropy, scaled_sum_eigen]).reshape(1, 8)
                                    # prediction
                                    prediction = pickle_model.predict(
                                        eigen_feature_array)

                                    # show predicted outputs
                                    for i in range(len(eigen_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} | frame : {}".format(
                                            prediction[i], predicted_class, detected_array.frame_number))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)

                                if(all_feature == True):
                                    # Load the scaler for all the features
                                    scaler_h = cPickle.load(
                                        open(selected_scaler_h, 'rb'))
                                    scaler_w = cPickle.load(
                                        open(selected_scaler_w, 'rb'))
                                    scaler_v = cPickle.load(
                                        open(selected_scaler_v, 'rb'))
                                    scaler_l1 = cPickle.load(
                                        open(selected_scaler_l1, 'rb'))
                                    scaler_l2 = cPickle.load(
                                        open(selected_scaler_l2, 'rb'))
                                    scaler_l3 = cPickle.load(
                                        open(selected_scaler_l3, 'rb'))
                                    scaler_eigen_curve = cPickle.load(
                                        open(selected_scaler_eigen_curve, 'rb'))
                                    scaler_eigen_entropy = cPickle.load(
                                        open(selected_scaler_eigen_entropy, 'rb'))
                                    scaler_omnivariance = cPickle.load(
                                        open(selected_scaler_omnivari, 'rb'))
                                    scaler_anisotropy = cPickle.load(
                                        open(selected_scaler_anisotr, 'rb'))
                                    scaler_sum_eigen = cPickle.load(
                                        open(selected_scaler_sum_eigen, 'rb'))
                                    # transform incoming features according to scaling
                                    scaled_h = scaler_h.transform(
                                        np.array(height).reshape(1, 1))
                                    scaled_w = scaler_w.transform(
                                        np.array(width).reshape(1, 1))
                                    scaled_v = scaler_v.transform(
                                        np.array(volume).reshape(1, 1))
                                    scaled_l1 = scaler_l1.transform(
                                        np.array(L1).reshape(1, 1))
                                    scaled_l2 = scaler_l2.transform(
                                        np.array(L2).reshape(1, 1))
                                    scaled_l3 = scaler_l3.transform(
                                        np.array(L3).reshape(1, 1))
                                    scaled_eigen_curve = scaler_eigen_curve.transform(
                                        np.array(eigen_curvature).reshape(1, 1))
                                    scaled_eigen_entropy = scaler_eigen_entropy.transform(
                                        np.array(eigen_entropy).reshape(1, 1))
                                    scaled_omnivariance = scaler_omnivariance.transform(
                                        np.array(omnivariance).reshape(1, 1))
                                    scaled_anisotropy = scaler_anisotropy.transform(
                                        np.array(anisotropy).reshape(1, 1))
                                    scaled_sum_eigen = scaler_sum_eigen.transform(
                                        np.array(sum_eigen).reshape(1, 1))

                                    # create feature array
                                    all_feature_array = np.array(
                                        [scaled_h, scaled_w, scaled_v, scaled_l1, scaled_l2, scaled_l3, scaled_eigen_curve, scaled_eigen_entropy, scaled_omnivariance, scaled_anisotropy, scaled_sum_eigen]).reshape(1, 11)

                                    # prediction
                                    prediction = pickle_model.predict(
                                        all_feature_array)

                                    # show predicted outputs
                                    for i in range(len(all_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} ".format(
                                            prediction[i], predicted_class))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class
                                    
                                    # publish
                                    #pub.publish(detected_array)

                            else:
                                if(simple_feature == True):
                                    simple_feature_array = np.array(
                                        [height, width, volume]).reshape(1, 3)
                                    # prediction
                                    prediction = pickle_model.predict(
                                        simple_feature_array)

                                    # show predicted outputs
                                    for i in range(len(simple_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} | frame : {}".format(
                                            prediction[i], predicted_class, detected_array.frame_number))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)

                                if(eigen_feature == True):
                                    eigen_feature_array = np.array(
                                        [L1, L2, L3, eigen_curvature, eigen_entropy, omnivariance, anisotropy, sum_eigen]).reshape(1, 8)
                                    # prediction
                                    prediction = pickle_model.predict(
                                        eigen_feature_array)

                                    # show predicted outputs
                                    for i in range(len(eigen_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} ".format(
                                            prediction[i], predicted_class))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)
                                if(all_feature == True):
                                    all_feature_array = np.array(
                                        [height, width, volume, L1, L2, L3, eigen_curvature, eigen_entropy, omnivariance, anisotropy, sum_eigen]).reshape(1, 11)
                                    # prediction
                                    prediction = pickle_model.predict(
                                        all_feature_array)

                                    # show predicted outputs
                                    for i in range(len(all_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} ".format(
                                            prediction[i], predicted_class))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)

            else:
                print("Invalid eigen features")
                # load the model from disk
                SVM_Path1 = model_path_with_scaling+'/svm/simple_features/'
                # selected trained model
                selected_model1 = SVM_Path1+'4_SVM_balanced_with_scaling_simple_features.pkl'
                # selected trained scaler per each feature
                selected_scaler_h1 = SVM_Path1+'4_SVM_scaler_h_simple.sav'
                selected_scaler_w1 = SVM_Path1+'4_SVM_scaler_w_simple.sav'
                selected_scaler_v1 = SVM_Path1+'4_SVM_scaler_v_simple.sav'
                simple_feature1 = True
                with open(selected_model1, 'rb') as file:
                    # pickled model from jupyter
                    pickle_model = cPickle.load(file)

                    if pickle_model:
                        # feature scaling
                        if(feature_scaling == True):
                            if(simple_feature1 == True):
                                # Load the scaler for all the features

                                scaler_h1 = cPickle.load(
                                    open(selected_scaler_h1, 'rb'))
                                scaler_w1 = cPickle.load(
                                    open(selected_scaler_w1, 'rb'))
                                scaler_v1 = cPickle.load(
                                    open(selected_scaler_v1, 'rb'))
                                # transform incoming features according to scaling
                                scaled_h1 = scaler_h1.transform(
                                    np.array(height).reshape(1, 1))
                                scaled_w1 = scaler_w1.transform(
                                    np.array(width).reshape(1, 1))
                                scaled_v1 = scaler_v1.transform(
                                    np.array(volume).reshape(1, 1))

                                # create feature array
                                simple_feature_array = np.array(
                                    [scaled_h1, scaled_w1, scaled_v1]).reshape(1, 3)
                                # prediction
                                prediction = pickle_model.predict(
                                    simple_feature_array)

                                # show predicted outputs

                                for i in range(len(simple_feature_array)):

                                    # reverse mapping of class labels from integer to string
                                    predicted_class = reverse_mapping(
                                        prediction[i])
                                    print("Prediction_class : {} | predicted_class : {} | frame : {}".format(
                                        prediction[i], predicted_class, detected_array.frame_number))
                                    # push predicted class to class_label_pred
                                    detected_array.objects[index].class_label_pred = predicted_class
                                    # publish
                                    #pub.publish(detected_array)

                            else:
                                if(simple_feature1 == True):
                                    simple_feature_array = np.array(
                                        [height, width, volume]).reshape(1, 3)
                                    # prediction
                                    prediction = pickle_model.predict(
                                        simple_feature_array)

                                    # show predicted outputs
                                    for i in range(len(simple_feature_array)):

                                        # reverse mapping of class labels from integer to string
                                        predicted_class = reverse_mapping(
                                            prediction[i])
                                        print("Prediction_class : {} | predicted_class : {} | frame : {}".format(
                                            prediction[i], predicted_class, detected_array.frame_number))
                                        # push predicted class to class_label_pred
                                        detected_array.objects[index].class_label_pred = predicted_class

    pub.publish(detected_array)                 

    print("----------------------------------------------")


def predict():

    rospy.init_node('online_class_prediction', anonymous=False)

    rospy.Subscriber("/detection/lidar_tracker/objects",
                     DetectedObjectArray, callback)
    rospy.spin()


if __name__ == '__main__':
    predict()
