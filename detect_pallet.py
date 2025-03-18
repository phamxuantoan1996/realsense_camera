import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import open3d as o3d
from utils import *

from flask import Flask, jsonify, request
from flask_cors import CORS

import time

ROI_X1 = 0
ROI_Y1 = 199
ROI_X2 = 640
ROI_Y2 = 383

PALLET_HEIGHT_THRES1 = 0.03
PALLET_HEIGHT_THRES2 = 0.06

PALLET_AXIS_Y_THRES1 = 80
PALLET_AXIS_Y_THRES2 = 100

PALLET_AXIS_Z_THRES1 = 0
PALLET_AXIS_Z_THRES2 = 30

TOTAL_POINT_MIN = 100
PALLET_POINT_MIN = 20

# app = Flask(__name__)
# CORS(app=app)
    
# @app.route('/detect_pallet',methods=['POST'])
# def enable_detect_pallet():
#     try:
#         return jsonify({"result":True,"desc":""}),201
#     except Exception as e:
#         return jsonify({"result":False,"desc":str(e)}),500
    
# @app.route('/detect_pallet',methods=['GET'])
# def result_detect_pallet():
#     try:
#         return jsonify({"ret_code":1}),200
#     except Exception as e:
#         return jsonify({"result":False,"desc":str(e)}),500

# if __name__ == '__main__':
#     app.run(host="0.0.0.0",port=8000,debug=False)

def calculate_rotation_angle(plane_model):
    """
    Tính toán góc xoay của mặt phẳng dựa trên vector pháp tuyến (A, B, C).
    Góc được tính so với các mặt phẳng XY, XZ, và YZ.
    """
    # Các thành phần của vector pháp tuyến (A, B, C)
    A, B, C, D = plane_model
    # Tính góc giữa vector pháp tuyến và các trục X, Y, Z
    normal_vector = np.array([A, B, C])
    # Trục X, Y, Z trong không gian 3D
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    # Tính góc giữa vector pháp tuyến và các trục
    def calculate_angle(normal_vector, axis):
        dot_product = np.dot(normal_vector, axis)
        norm_normal = np.linalg.norm(normal_vector)
        norm_axis = np.linalg.norm(axis)
        cos_theta = dot_product / (norm_normal * norm_axis)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Đảm bảo giá trị nằm trong [−1, 1]
        return np.degrees(theta)  # Chuyển sang độ
        # Tính các góc với trục X, Y, Z
    angle_with_x = calculate_angle(normal_vector, x_axis)
    angle_with_y = calculate_angle(normal_vector, y_axis)
    angle_with_z = calculate_angle(normal_vector, z_axis)
    return angle_with_x, angle_with_y, angle_with_z

def ransac_plane_segmentation(point_cloud):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # Segment plane using RANSAC
    plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=0.0015,ransac_n=3,num_iterations=1000)
    # Chọn các điểm inliers thuộc mặt phẳng
    inlier_cloud = o3d_cloud.select_by_index(inliers)
    # Chọn các điểm outliers thuộc mặt phẳng
    outlier_cloud = o3d_cloud.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud
    
def DetectMultiPlanes(points):
    plane_list = []
    temp = points
    while True:
        plane_model,inlier_cloud,outlier_cloud = ransac_plane_segmentation(temp)
        angleX,angleY,angleZ = calculate_rotation_angle(plane_model)
        if np.asarray(inlier_cloud.points).size/3 > PALLET_POINT_MIN and (angleY > PALLET_AXIS_Y_THRES1 and angleY < PALLET_AXIS_Y_THRES2) and (angleZ > PALLET_AXIS_Z_THRES1 and angleZ < PALLET_AXIS_Z_THRES2):
            plane_list.append((plane_model,inlier_cloud,angleX,angleY,angleZ))
        if np.asarray(outlier_cloud.points).size/3 < TOTAL_POINT_MIN:
            break
        temp = PCDToNumpy(outlier_cloud)
    return plane_list

def detect_obstacle() -> bool:
    pipeline.start(config)
    for i in range(0,10):
        frames = pipeline.wait_for_frames()
    try:
        obstacle = True
        index = 0
        pcd_total = None
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image
            aligned_frames = align.process(frames)
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            depth_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            points = []
            for i in range(ROI_Y1, ROI_Y2):
                for j in range(ROI_X1, ROI_X2):
                    if i < color_image.shape[0] and j < color_image.shape[1]:# Kiểm tra xem pixel có trong ROI không
                        dist = depth_frame.get_distance(j, i)  # Dùng depth_frame gốc ở đây
                        if dist > 0:  # Nếu có điểm hợp lệ
                            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [j, i], dist)
                            if (point[1] > 0.01 and point[1] < 0.07) and (point[2] > 0.3 and point[2] < 1.0) and (point[0] > -0.2 and point[0] < 0.2):
                                points.append(point)
            points_remove_noise = RemoveNoiseStatistical(points,nb_neighbors=50, std_ratio=0.01)
            
            pcd = NumpyToPCD(points_remove_noise)
            if pcd_total == None:
                pcd_total = pcd
            else:
                pcd_total = pcd_total + pcd
    
            index = index + 1
            print(index)
            if index < 20:
                continue

            vis = o3d.visualization.Visualizer()
            vis.create_window("render")
            vis.add_geometry(pcd_total)

            vis.run()
            vis.destroy_window()
            break

    except Exception as e:
        print(e)
    pipeline.stop()
    return obstacle


def detect_pallet_deflection() -> bool:
    pipeline.start(config)
    for i in range(0,10):
        frames = pipeline.wait_for_frames()
    try:
        deflection = True
        index = 0
        pcd_total = None
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image
            aligned_frames = align.process(frames)
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            depth_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            # Validate that both frames are valid
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            points = []
            # Lọc các điểm trong ROI
            for i in range(ROI_Y1, ROI_Y2):
                for j in range(ROI_X1, ROI_X2):
                    if i < color_image.shape[0] and j < color_image.shape[1]:# Kiểm tra xem pixel có trong ROI không
                        dist = depth_frame.get_distance(j, i)  # Dùng depth_frame gốc ở đây
                        if dist > 0:  # Nếu có điểm hợp lệ
                            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [j, i], dist)
                            if point[1] > PALLET_HEIGHT_THRES1 and point[1] < PALLET_HEIGHT_THRES2 and point[2] > 0 and point[2] < 0.6:
                                points.append(point)
            points_remove_noise = RemoveNoiseStatistical(points,nb_neighbors=50, std_ratio=0.01)
            points_sample_down = DownSample(points_remove_noise,voxel_size=0.01)
            pcd = NumpyToPCD(points_sample_down)
            if pcd_total == None:
                pcd_total = pcd
            else:
                pcd_total = pcd_total + pcd
    
            index = index + 1
            # print(index)
            if index < 50:
                continue

            temp = PCDToNumpy(pcd_total)
            temp_remove_noise = RemoveNoiseStatistical(temp,nb_neighbors=30, std_ratio=0.01)
            temp_down_sample = DownSample(temp_remove_noise,voxel_size=0.01)
            
            vis = o3d.visualization.Visualizer()
            vis.create_window("render")
            vis.add_geometry(pcd_total)
            
            plane_list = DetectMultiPlanes(temp_down_sample)
            # print(len(plane_list))
            if len(plane_list) == 1:
                for item in plane_list:
                    inliers = item[1]
                    inliers.paint_uniform_color([0, 0, 0])
                    aabb = inliers.get_axis_aligned_bounding_box()
                    aabb.color = (1, 0, 0)

                    vis.add_geometry(aabb)
                    vis.add_geometry(inliers)
                    angleZ = item[4]
                    center = aabb.get_center()
                    shape = aabb.get_extent()
                    print('center: ',center[0])
                    print('distance : ',center[2])
                    print('shape',shape[0])
                    print('angle Z : ',angleZ)
                    if round(angleZ,0) > 3 or round(shape[0],2) > 0.15 or round(center[0],3) > 0.025 or round(center[0],3) < -0.025:
                        deflection = True
                        print('lech')
                    else:
                        deflection = False
                        print('khong lech')
            print('-----------------------------------')
            vis.run()
            vis.destroy_window()
            break
    except Exception as e:
        print(e)
    pipeline.stop()
    return deflection

if __name__ == '__main__':
    

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('105322250851')
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)

    
    for i in range(0,50):
        detect_pallet_deflection()
        # detect_obstacle()
        time.sleep(5)
    
    

    



    
