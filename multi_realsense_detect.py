import pyrealsense2 as rs
import numpy as np
import cv2 as cv

from ultralytics import YOLO
from threading import Thread

from queue import Queue
import time

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
            return True
        except Exception as e:
            print(e)
            return False
    def realsense_stop(self):
        self.pipeline.stop()

def task_capture_frames_func(realsenses:list,queue_frames:Queue):
    while True:
        try:
            list_detect = []
            for realsense in realsenses:
                frame = realsense.pipeline.wait_for_frames()
                # Align the depth frame to color frame
                aligned_frame = realsense.align.process(frame)
                # Get aligned frames
                aligned_depth_frame = aligned_frame.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frame.get_color_frame()
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                rot_color = cv.rotate(color_image,cv.ROTATE_90_COUNTERCLOCKWISE)
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                rot_depth = cv.rotate(depth_image,cv.ROTATE_90_COUNTERCLOCKWISE)
                element = (rot_color,aligned_depth_frame,rot_depth)
                list_detect.append(element)
            
            img_v = None
            depth_v = None
                
            if len(list_detect) == 2:
                img_v = cv.hconcat([list_detect[0][0],list_detect[1][0]])
                depth_v = cv.hconcat([list_detect[0][2],list_detect[1][2]])
            elif len(list_detect) == 1:
                img_v = cv.hconcat([list_detect[0][0],list_detect[0][0]])
                depth_v = cv.hconcat([list_detect[0][2],list_detect[0][2]])

            img_resize = cv.resize(img_v,dsize=None,fx=0.5,fy=0.5) #640,960

            model = YOLO("yolo11n-seg.pt")
            results = model.predict(source=img_resize, save=False, classes=[0],imgsz=[320,480])

            
            # arr_np = np.zeros(shape=(960,640),dtype=np.uint8)
            # img_person = arr_np.reshape(960,640)
            # for result in results:
            #     masks = result.masks
            #     if masks != None:
            #         for mask in masks:
            #             mask_person = (mask.data[0].numpy())*255
            #             person = mask_person.astype(np.uint8)
            #             tmp = cv.bitwise_or(img_person,person)
            #             img_person = tmp

            # m = img_person == 0
            # dp = np.ma.masked_array(depth_v,m)
            # minval = np.mean(dp[np.nonzero(dp)])*(list_detect[0][1].get_units())
            # print('distance : ',minval)
            # cv.putText(img_v, str(minval), (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = f"{class_name} {confidence:.2f}"
                    if class_name == "person":          
                        cv.rectangle(img_resize, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            img_scale = cv.resize(img_resize,dsize=None,fx=2,fy=2)
            

            cv.imshow('win',img_scale)
            if cv.waitKey(1) == ord('q'):
                break
        except Exception as e:
            print(e)
    realsense1.realsense_stop()
    # realsense2.realsense_stop()






if __name__ == '__main__':
    queue_frames = Queue(maxsize=10240)
    realsense1 = realsense(serial_number='105322250851')
    realsense1.realsense_config()
    # realsense2 = realsense(serial_number='213522072335')
    # realsense2.realsense_config()

    task_capture_frames = Thread(target=task_capture_frames_func,args=(realsenses:=[realsense1],queue_frames:=queue_frames))
    task_capture_frames.start()

    

    



    
    

    