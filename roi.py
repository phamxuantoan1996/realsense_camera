import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        # This call waits until a new coherent set of frames is available on a device
        frame = pipeline.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())



        cv2.imshow("window",color_image)
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(e)
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()