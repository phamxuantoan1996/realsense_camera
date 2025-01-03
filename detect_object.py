import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO
import math
from PIL import Image
import torch



def check_distance_human(roi,rgb_image,depth_frame,color_intrin) -> bool:
    x1, y1, x2, y2 = roi
    centerx = x1 + round((x2-x1)/2)
    centery = y1 + round((y2-y1)/2)
    cv2.circle(rgb_image,(centerx,centery),2,(0,0,255),2)

    depth = depth_frame.get_distance(centerx, centery)
    dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [centerx,centery], depth)
    distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
    print('distance : ',distance)
    if distance > 1.0:
        return True
    
    return False

def camera_config():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    #######################################################################

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return align,pipeline

def check_human(align,pipeline):
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            
            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                


            # model = YOLO("yolo11n-seg.pt")
            # results = model.track(color_image)
            # for result in results:
            #     boxes = result.boxes
            #     masks = result.masks
            #     ori_img = result.orig_img
            #     for box in boxes:
            #         class_id = int(box.cls[0])
            #         class_name = model.names[class_id]
            #         x1, y1, x2, y2 = box.xyxy[0]
            #         confidence = box.conf[0]
            #         label = f"{class_name} {confidence:.2f}"
            #         if class_name == 'person':
            #             mask = masks[class_id].data[0].numpy()
            #             print(class_id)
            #             print(class_name)
            #             cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            #             cv2.putText(color_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #             cv2.imshow('mask',mask)
            

            # model = YOLO("yolo11n-seg.pt")
            # results = model.track(color_image)
            # for result in results:
            #     boxes = result.boxes
            #     for box in boxes:
            #         confidence = box.conf[0]
            #         # Get the class ID and name
            #         class_id = int(box.cls[0])
            #         class_name = model.names[class_id]
            #         x1, y1, x2, y2 = box.xyxy[0]
            #         label = f"{class_name} {confidence:.2f}"
            #         # Bước 2 :Lọc nguoi
            #         if class_name == "person":
                        
            #             roi = (int(x1),int(y1),int(x2),int(y2))
            #             if check_distance_human(roi,color_image,aligned_depth_frame,color_intrin):

            #                 cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            #                 cv2.putText(color_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #             else:
            #                 cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            #                 cv2.putText(color_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('window1',color_image)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

if __name__ == "__main__":
    align,pipeline = camera_config()
    check_human(align,pipeline)


