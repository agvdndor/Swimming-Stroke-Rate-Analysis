import argparse
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import time
from collections import deque
import sys
from operator import xor

global colors
colors = {'yellow': [(23, 100, 0), (30, 255, 255)],
          'blue': [(100, 150, 0),(120, 255, 255)],
          'orange':[(10,225,0),(15,255,255)],
          'tennisball': [(20,70,0),(35,255,255)],
          'turquoise': [(75,70,0),(95,255,255)],
          'null': [(0,0,0),(0,0,0)]
          }
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def callback(value):
    pass

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--filter', required=True,
                    help='Range filter. RGB or HSV')
    ap.add_argument('-v', '--video', required=False,
                    help='Path to the image')
    ap.add_argument('-w', '--webcam', required=False,
                    help='Use webcam', action='store_true')
    ap.add_argument('-p', '--preview', required=False,
                    help='Show a preview of the image after applying the mask',
                    action='store_true')
    args = vars(ap.parse_args())

    if not args['filter'].upper() in ['RGB', 'HSV']:
        ap.error("Please speciy a correct filter.")

    return args

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if i == "MIN" else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

def main():
    args = get_arguments()
    if args['video']:
            vs = cv2.VideoCapture(args["video"])
    else:
        print("please provide video")

    range_filter = args['filter'].upper()


    setup_trackbars(range_filter)

    while True:
        ret, image = vs.read()
        
        
        while True:
            if image is None:
                break
            if range_filter == 'RGB':
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

            thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

            mask = cv2.erode(thresh, None, iterations=2)
            mask = cv2.dilate(mask, kernel, iterations=2)

            if args['preview']:
                preview = cv2.bitwise_and(image, image, mask=mask)
                preview = ResizeWithAspectRatio(preview, width=1280)
                cv2.imshow("Preview", preview)
            else:
                cv2.imshow("Original", image)
                cv2.imshow("Thresh", mask)

            if cv2.waitKey(1) & 0xFF is ord('n'):
                break
        


if __name__ == '__main__':
    main()

