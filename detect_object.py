import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
from ultralytics import YOLO

def camera_config():
    # Create a pipeline
    pipeline = rs.pipeline()
    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    # Get device product line for setting a supporting resolution
    # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    #######################################################################
    # found_rgb = False
    # for s in device.sensors:
    #     if s.get_info(rs.camera_info.name) == 'RGB Camera':
    #         found_rgb = True
    #         break
    # if not found_rgb:
    #     print("The demo requires Depth camera with Color sensor")
    #     exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: " , depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return align,pipeline

def check_human(align,pipeline,thres = 1.0) -> bool:
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

            # depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                
            model = YOLO("yolo11n-seg.pt")
            results = model.predict(source=color_image, save=False, classes=[0])
            arr_np = np.zeros(shape=(480,640),dtype=np.uint8)
            img_person = arr_np.reshape(480,640)
            for result in results:
                masks = result.masks
                if masks != None:
                    for mask in masks:
                        mask_person = (mask.data[0].numpy())*255
                        person = mask_person.astype(np.uint8)
                        tmp = cv2.bitwise_or(img_person,person)
                        img_person = tmp
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = f"{class_name} {confidence:.2f}"
                    if class_name == "person":          
                        point_cloud = []
                        for x in range(int(x1),int(x2)):
                            for y in range(int(y1),int(y2)):
                                if img_person[y,x] != 0:
                                    if aligned_depth_frame.get_distance(x,y) > 0:
                                        point_cloud.append(aligned_depth_frame.get_distance(x,y))
                        if len(point_cloud) > 0:
                            dis = min(point_cloud)
                            dis = round(dis,4)
                            if dis > thres:
                                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            else:
                                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(color_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.putText(color_image, str(dis), (int(x1), int(y1) + int((int(y2) - int(y1))/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # cv2.imshow('seg',img_person)
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
    check_human(align,pipeline,thres=0.5)


