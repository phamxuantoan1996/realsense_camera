import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from ultralytics import YOLO

from threading import Thread
import time

import open3d as o3d

from threading import Thread
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app=app)

class realsense:
    def __init__(self,serial_number:str):
        self.serial_number = serial_number
    
    def realsense_config(self):
        # Create a pipeline
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(config)

            align_to = rs.stream.color
            self.align = rs.align(align_to)

            depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.depth_units, 0.0001)  # Depth units in meters
            depth_sensor.set_option(rs.option.visual_preset, 3)  # Use visual preset for high accuracy
            return True
        except Exception as e:
            print(e)
            return False
    def realsense_stop(self):
        self.pipeline.stop()

def apply_roi_to_pointcloud(roi, rgb_image, depth_frame, depth_intrinsics):
    """
    Áp dụng ROI chọn được lên point cloud từ RGB và độ sâu.
    """
    x, y, w, h = roi
    roi_points = []
    # Lọc các điểm trong ROI
    for i in range(y, y + h):
        for j in range(x, x + w):
            if i < rgb_image.shape[0] and j < rgb_image.shape[1]:
                # Kiểm tra xem pixel có trong ROI không
                dist = depth_frame.get_distance(j, i)  # Dùng depth_frame gốc ở đây
                if dist > 0:  # Nếu có điểm hợp lệ
                    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [j, i], dist)
                    roi_points.append(point)
    return np.array(roi_points)

def ransac_plane_segmentation(point_cloud):
    """
    Áp dụng RANSAC để tách mặt phẳng khỏi point cloud và tính diện tích mặt phẳng.
    """
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)

    # Segment plane using RANSAC
    plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=0.01,
                                                   ransac_n=3,
                                                   num_iterations=1000)
    # print("Mặt phẳng được tìm thấy:", plane_model)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    # Chọn các điểm inliers thuộc mặt phẳng
    inlier_cloud = o3d_cloud.select_by_index(inliers)
    points_center = np.asarray(inlier_cloud.points)
    center = np.mean(points_center, axis=0)
    # print(f"Tâm mặt phẳng: {center}")
    
    # Tính diện tích mặt phẳng bằng cách sử dụng convex hull
    hull, _ = inlier_cloud.compute_convex_hull()
    hull_area = hull.get_surface_area()  # Diện tích của bao lồi (convex hull)
    
    # print(f"Diện tích mặt phẳng: {hull_area:.2f} m²")
    
    # Lấy các điểm và tách mặt phẳng và các điểm còn lại
    outlier_cloud = o3d_cloud.select_by_index(inliers, invert=True)
    
    return plane_model, inlier_cloud, outlier_cloud, hull_area, center

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


def calculate_angle_with_oyz(plane_model):
    """
    Tính góc giữa mặt phẳng bất kỳ và mặt phẳng Oyz.
    """
    A, B, C, _ = plane_model
    normal_vector = np.array([A, B, C])
    oyz_normal = np.array([1, 0, 0])  # Vector pháp tuyến của mặt phẳng Oyz
    cos_phi = np.dot(normal_vector, oyz_normal) / (np.linalg.norm(normal_vector) * np.linalg.norm(oyz_normal))
    phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))  # Đảm bảo cos_phi nằm trong [-1, 1]
    return np.degrees(phi)

def calib_array_value(arr_np:np.ndarray) -> np.ndarray:
    mean1 = arr_np.mean()
    arr_np1 = arr_np[np.where(arr_np > mean1)]

    mean2 = arr_np.mean()
    arr_np2 = arr_np[np.where(arr_np < mean2)]

    if arr_np1.shape > arr_np2.shape:
        return arr_np1
    elif arr_np1.shape < arr_np2.shape:
        return arr_np2
    else:
        return None

def task_detect_pallet_func(realcamera:realsense,serial:str):
    model = YOLO("runs/detect/train29/weights/best.pt")
    while True:
        if not detect_pallet["enable"]:
            time.sleep(1)
            continue
        list_delta_x = []
        list_delta_z = []
        list_angle = []
        detect_pallet["status"] = 'active'
        for i in range(0,25):
            try:
                frame = realcamera.pipeline.wait_for_frames()
                # Align the depth frame to color frame
                aligned_frame = realcamera.align.process(frame)
                # Get aligned frames
                aligned_depth_frame = aligned_frame.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frame.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                depth_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                
                results = model.track(color_image)
                confidence_max = 0
                box_max = None
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        confidence = box.conf[0]
                        # Get the class ID and name
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        # Bước 2 :Lọc pallet
                        if class_name == "pallet":
                            if confidence_max < confidence:
                                confidence_max = confidence
                                box_max = box
                if box_max != None:
                    x1, y1, x2, y2 = box_max.xyxy[0]
                    label = f"{class_name} {confidence_max:.2f}"
                    roi = (int(x1),int(y1),int(x2),int(y2))
                    roi_points = apply_roi_to_pointcloud(roi=roi,rgb_image=color_image,depth_frame=aligned_depth_frame,depth_intrinsics=depth_intrinsics)
                    try:
                        plane_model, plane, edges, hull_area,center = ransac_plane_segmentation(roi_points)
                        angle_x,angle_y,angle_z = calculate_rotation_angle(plane_model)
                        # angle = calculate_rotation_angle(plane_model)
                        # print('delta angle : ',angle)
                        print(f"Góc xoay của mặt phẳng so với mặt phẳng ngang (XY plane): {angle_z:.2f}°")
                        cv.putText(color_image, f"ROI: {int(x1), int(y1), int(x2), int(y2)}", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv.putText(color_image, f"Pose: {center}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv.putText(color_image, f"Angle: {angle_z:.2f}", (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv.putText(color_image, f"S: {hull_area:.2f}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        list_delta_x.append(float(center[0]))
                        list_delta_z.append(float(center[2]))
                        list_angle.append(float(angle_z))
                        
                    except Exception as e:
                        print(e)

                    cv.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv.putText(color_image, label, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv.imshow('win1',color_image)
                if cv.waitKey(1) == ord('q'):
                    pass
            except Exception as e:
                print(e)
        cv.destroyAllWindows()


        arr_angle = np.array(list_angle)
        calib_arr_angle = calib_array_value(arr_angle)
        detect_pallet["result"]["angle"] = calib_arr_angle.mean()
        # mean_angle = arr_angle.mean()
        # mask_angle = arr_angle > mean_angle
        # detect_pallet["result"]["angle"] = np.ma.masked_array(arr_angle,mask_angle).mean()


        arr_z = np.array(list_delta_z)
        calib_arr_z = calib_array_value(arr_z)
        detect_pallet["result"]["angle"] = calib_arr_z.mean()
        # detect_pallet["result"]["delta_z"] = arr_z.mean()

        arr_x = np.array(list_delta_x)
        calib_arr_x = calib_array_value(arr_x)
        detect_pallet["result"]["delta_x"] = calib_arr_x.mean()
        # mean_x = arr_x.mean()
        # mask_x = arr_x > mean_x
        # detect_pallet["result"]["delta_x"] = np.ma.masked_array(arr_x,mask_x).mean()

        detect_pallet["status"] = "complete"
        detect_pallet["enable"] = False
    
    
@app.route('/detect_pallet',methods=['POST'])
def enable_detect_pallet():
    try:
        content = request.json
        keys = content.keys()
        if 'enable' in keys:
            detect_pallet['enable'] = content['enable']
            return jsonify({"result":True,"desc":""}),201
        
        return jsonify({"result":False,"desc":""}),200
    except Exception as e:
        return jsonify({"result":False,"desc":str(e)}),500
    
@app.route('/detect_pallet',methods=['GET'])
def result_detect_pallet():
    try:
        if detect_pallet["status"] == 'complete':
            detect_pallet["status"] = "deactive"
            return jsonify(detect_pallet),200
        else:
            return jsonify({"result":False,"desc":""}),404
    except Exception as e:
        return jsonify({"result":False,"desc":str(e)}),500

if __name__ == '__main__':
    detect_pallet = {
        "status": "deactive",
        "enable":False,
        "result":{
            "angle":0,
            "delta_z":0,
            "delta_x":0
        }
    }

    realsense1 = realsense(serial_number='105322250851')
    print(realsense1.realsense_config())

    task_detect_pallet = Thread(target=task_detect_pallet_func,args=(realcamera:=realsense1,serial:='123'))
    task_detect_pallet.start()

    task_server = Thread(target=app.run,args=(host:='0.0.0.0',port:=8002))
    task_server.start()



    
