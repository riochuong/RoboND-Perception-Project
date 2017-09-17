#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    threshold_filter = 0.5
    outlier_filter.set_std_dev_mul_thresh(threshold_filter)
    clould_filtered = outlier_filter.filter()

    # TODO: Voxel Grid Downsampling
    LEAF_SIZE = 0.01
    #vox_filter = pcl_data.make_voxel_grid_filter()
    vox_filter = clould_filtered.make_voxel_grid_filter()
    vox_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    clould_filtered = vox_filter.filter()

    # TODO: PassThrough Filter
    filter_axix = 'z'
    passthrough_table = clould_filtered.make_passthrough_filter()
    passthrough_table.set_filter_field_name(filter_axix)

    # set limits for table and object
    passthrough_table.set_filter_limits(0.6,1.5)
    table_filter = passthrough_table.filter()

    # set limit for y axis
    filter_y = 'y'
    limit_y_table = table_filter.make_passthrough_filter()
    limit_y_table.set_filter_field_name(filter_y)
    limit_y_table.set_filter_limits(-0.5,0.5)
    table_filter = limit_y_table.filter()


    # TODO: RANSAC Plane Segmentation
    segmenter = table_filter.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    segmenter.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliners, coefficients = segmenter.segment()
    table_inliers = table_filter.extract(inliners, negative=False)
    object_outliers =  table_filter.extract(inliners, negative=True)

    # TODO: Euclidean Clustering
    # this function will perform DBSCAN
    white_cloud = XYZRGB_to_XYZ(object_outliers)
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # create cluster extraction object 
    euclidean_cluster = white_cloud.make_EuclideanClusterExtraction()

    # set cluster tolerance info
    # TODO : NEED SOME EXPERIMENTS here
    euclidean_cluster.set_ClusterTolerance(0.01)
    euclidean_cluster.set_MinClusterSize(100)
    euclidean_cluster.set_MaxClusterSize(1000)

    # search kd tree
    euclidean_cluster.set_SearchMethod(tree)
    cluster_indices = euclidean_cluster.Extract()

    # TODO: Convert PCL data to ROS messages
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(object_outliers)
    ros_cloud_table =  pcl_to_ros(table_inliers)
    ros_cluster = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    # pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster)


# Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []
    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        print ("Classifying ",index)
        pcl_cluster = object_outliers.extract(pts_list)
        
        # Compute the associated feature vector
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
    
        # TODO: complete this step just as is covered in capture_features.py
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        print("Detected object ",index," is [",label,"]")

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

     # Publish the list of detected objects
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    detected_objects_pub.publish(detected_objects)
   
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    centroids = {}
    test_scene_num = Int32()
    test_scene_num.data = 1
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')

    # TODO: Parse parameters into individual variables
    for detected_object in object_list:
        label = detected_object.label
        # convert object cloud to array
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        # get the centroids location 
        centroid = np.mean(points_arr, axis=0)[:3]
        centroids[label] = centroid

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    dict_list = []
    for i in range(0,len(object_list_param)):
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        label = object_list_param[i]['name']
        centroid = centroids[label]
        group = object_list_param[i]['group']
        object_name = String()
        object_name.data = label
        pick_pose = Pose()
        place_pose = Pose()
        arm_name = String()

        # based on observation left=red, green=right 
        if group == "red":
            arm_name.data = "left"
            place_pose.position.x = 0
            place_pose.position.y = 0.71
            place_pose.position.z = 0.605
        else:
            arm_name.data = "right"
            place_pose.position.x = 0
            place_pose.position.y = -0.71
            place_pose.position.z = 0.605

        # assign value for pose
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])

        print("centroid of ",label," is ",centroid)
        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        # try:
        #     #pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     #resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

        #     #print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml("test3_world.yaml", dict_list)



if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler  = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
