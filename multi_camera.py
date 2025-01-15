import pyrealsense2 as rs
import cv2 as cv
import numpy as np

# config camera1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('105322250851')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from both cameras
pipeline_1.start(config_1)

try:
    while True:
        frames_1 = pipeline_1.wait_for_frames()
        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        if not depth_frame_1 or not color_frame_1:
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())

        img_rot = cv.rotate(color_image_1,cv.ROTATE_90_COUNTERCLOCKWISE)

        img_v = cv.hconcat([img_rot,img_rot])

        cv.imshow('cam1',img_v)
        if cv.waitKey(1) == ord('q'):
            break

finally:
    pipeline_1.stop()