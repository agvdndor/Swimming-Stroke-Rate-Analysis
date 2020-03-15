import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
from range_detector_video import ResizeWithAspectRatio

def get_avg(video, stride, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max):
    vid_stream = cv2.VideoCapture(video)

    #get first frame
    ret, frame = vid_stream.read()

    # init with zeros
    num_frame = 0
    total = np.zeros(frame.shape[:2], dtype='int64')

    while ret:
        # get mask
        frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        # add current frame mask to total
        total = np.add(total, thresh/255)

        # update count
        num_frame += 1
        print(num_frame)
        for i in range(stride):
            ret, frame = vid_stream.read()
            if not ret:
                break
        ret, frame = vid_stream.read()
    print(total)
    avg = np.true_divide(total,num_frame)
    thresh = cv2.threshold(avg, 0.25, 1.0, cv2.THRESH_BINARY)[1]
    thresh = thresh * 255
    thresh = thresh.astype('uint8')
    print(thresh)
    return thresh



    print(total)
    return 'ok'


if __name__=='__main__':
    video = 'stitched.avi'
    v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = 0,0,0,80,120,255
    cv2.imshow("background", ResizeWithAspectRatio(get_avg(video, 50, v1_min, v2_min, v3_min, v1_max, v2_max, v3_max),width=1280))
    while True:
        if cv2.waitKey(1) & 0xFF is ord('n'):
                break