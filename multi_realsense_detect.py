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

def new_coordinates_after_resize_img(original_size, new_size, original_coordinate):
  original_size = np.array(original_size)
  new_size = np.array(new_size)
  original_coordinate = np.array(original_coordinate)
  xy = original_coordinate/(original_size/new_size)
  x, y = int(xy[0]), int(xy[1])
  return (x, y)

def task_capture_frames_func(realsenses:list,queue_frames:Queue):
    while True:
        try:
            list_detect = []
            index = 0
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
                depth_image = np.asanyarray(aligned_depth_frame.get_data())

                if index == 0:
                    rot_color = cv.rotate(color_image,cv.ROTATE_90_CLOCKWISE)
                    rot_depth = cv.rotate(depth_image,cv.ROTATE_90_CLOCKWISE)
                elif index == 1:
                    rot_color = cv.rotate(color_image,cv.ROTATE_90_COUNTERCLOCKWISE)
                    rot_depth = cv.rotate(depth_image,cv.ROTATE_90_COUNTERCLOCKWISE)

                element = (rot_color,aligned_depth_frame,rot_depth)
                list_detect.append(element)
                index = index + 1
            
            img_h = None
            depth_h = None
                
            if len(list_detect) == 2:
                img_h = cv.hconcat([list_detect[0][0],list_detect[1][0]])
                depth_h = cv.hconcat([list_detect[0][2],list_detect[1][2]])
            elif len(list_detect) == 1:
                img_h = cv.hconcat([list_detect[0][0],list_detect[0][0]])
                depth_h = cv.hconcat([list_detect[0][2],list_detect[0][2]])
            img_resize_dwn = cv.resize(img_h,dsize=None,fx=0.5,fy=0.5,interpolation=cv.INTER_CUBIC) #640,960

            model = YOLO("yolo11n-seg.pt")
            results = model.predict(source=img_resize_dwn, save=False, classes=[0],imgsz=[320,480])

            arr_np = np.zeros(shape=(320,480),dtype=np.uint8)
            img_person_mask = arr_np.reshape(320,480)
            for result in results:
                masks = result.masks
                if masks != None:
                    for mask in masks:
                        mask_person = (mask.data[0].numpy())*255
                        person = mask_person.astype(np.uint8)
                        tmp = cv.bitwise_or(img_person_mask,person)
                        img_person_mask = tmp

            img_person_mask_rz_up = cv.resize(img_person_mask,dsize=None,fx=2,fy=2,interpolation=cv.INTER_CUBIC)
            person_mask = img_person_mask_rz_up == 0
            depth_person = np.ma.masked_array(depth_h,person_mask)

            list_rec = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = f"{class_name} {confidence:.2f}"
                    if class_name == "person":          
                        list_rec.append((int(x1),int(y1),int(x2),int(y2)))
            
            if len(list_rec) != 0:
                print(len(list_rec))
                for rec in list_rec:
                    point1 = new_coordinates_after_resize_img((320,480), (640,960), (rec[0],rec[1]))
                    point2 = new_coordinates_after_resize_img((320,480), (640,960), (rec[2],rec[3]))
                    roi = depth_person[point1[1]:point2[1],point1[0]:point2[0]]
                    distance = np.mean(roi[np.nonzero(roi)])*(list_detect[0][1].get_units())
                    distance = round(distance,4)
                    cv.putText(img_h, str(distance), (point1[0]+5, point1[1]+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                    cv.rectangle(img_h,point1,point2,(0,0,255),2)
            
            

            cv.imshow('win1',img_h)
            
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
    realsense2 = realsense(serial_number='213522072335')
    realsense2.realsense_config()

    task_capture_frames = Thread(target=task_capture_frames_func,args=(realsenses:=[realsense1,realsense2],queue_frames:=queue_frames))
    task_capture_frames.start()

    

    



    
    

    