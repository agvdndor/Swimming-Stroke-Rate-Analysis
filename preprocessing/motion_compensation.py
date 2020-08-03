import cv2
import numpy as np
import argparse
import sys
from avg_mask import get_avg
from tqdm import tqdm
import os
import csv

#global
#bg_subtractor = cv2.createBackgroundSubtractorKNN()
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = 0,0,0,80,120,255

#number of frames during warmup period
skip_frames = 100

# interval width and step
interval_width = 1000
interval_step = 10
interval_margin = 150
forward_margin = 200
backward_margin = 100

#output_dir = "C:\Users\Vande\data_preprocessed"


# dilation kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', required=True,
                    help='Path to the video')
    ap.add_argument('-d', '--debug', required=False, action='store_true',
                    help='Show debug views such as roi, foreground and step through video frame per frame')
    ap.add_argument('-w', '--display_width', type=int, default=1280)
    ap.add_argument('-f', '--save', required=False)
    ap.add_argument('--save_imgs', required=False, action='store_true')
    args = vars(ap.parse_args())

    return args

def video_frames(video):
    print('skipping frames...')
    vid = cv2.VideoCapture(video)
    while True:
        ret, frame = vid.read()
        if ret:
            yield frame 
        else:
            print('generator break')
            break

# mask to ignore upper and lower band of image
def get_roi(width, height, top_offset, bot_offset):
    top = np.zeros((top_offset, width), dtype='uint8') 
    bot = np.zeros((bot_offset, width), dtype='uint8')
    mid = np.ones((height - top_offset - bot_offset, width), dtype='uint8') * 255

    roi = np.append(top, mid, axis=0)
    roi = np.append(roi, bot, axis=0)
    return roi

def apply_mask(input, mask):
    return cv2.bitwise_and(input, mask)

def get_foreground(frame):
    return bg_subtractor.apply(frame)

def get_color_mask(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(frame_hsv, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))
    return thresh

def skip(gen, num_frames):
    for frame in tqdm(range(num_frames)):
        frame = gen.__next__()
        # let background stabilize
        get_foreground(frame)

def erode_and_dilate(frame):
    frame = cv2.erode(frame, kernel, iterations=2)
    frame = cv2.dilate(frame, kernel, iterations=2)
    return frame

def get_best_interval(mask, width):
    max = 0
    opt_offset = 0
    for i in range(0,width - interval_width + 1,interval_step):
        start = i
        stop = i + interval_width
        
        count = cv2.countNonZero(mask[:,start:stop])
        if count > max:
            max = count
            opt_offset = start

    return opt_offset 

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
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

def main():
    # get arguments
    args = get_arguments()

    output_dir = os.path.join(args['video'], '..', '..', "images")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get avg_mask to subtract
    # avg_mask get_avg(args['video'], 25 ,v1_min, v2_min, v3_min, v1_max, v2_max, v3_max)

    # get frame generator
    frame_gen = video_frames(args['video'])

    # get first frame
    frame = frame_gen.__next__()

    # get width and height
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]        

    num_history = 100
    position_history = np.zeros(num_history)

    # create videowriter if video is supposed to be saved
    if args['save']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fourcc = cv2.VideoWriter_fourcc(*'MPG4')
        writer = cv2.VideoWriter(os.path.join(output_dir, args['video'].split('.')[0], args['save']) , fourcc, 25.0, (interval_width + 2* interval_margin,frame_height))

    # register csv dialect
    csv.register_dialect('no_termination', delimiter = ',', lineterminator = '\n')

    roi = get_roi(frame_width, frame_height, 120, 150)

    frame_iter = 0
    direction = 0
    for frame in frame_gen:
        if frame_iter == 0 :
            # skip background
            print("warming up...")
            skip = True
        if frame_iter == skip_frames:
            print("warmup done")
            skip = False
        if frame_iter % 100== 0:
            print('frame index: ', frame_iter)

        cv2.imshow('frame', resize_with_aspect_ratio(frame, width=1280))
        
        foreground = get_foreground(frame)
        color_mask = get_color_mask(frame)
        #print('1')
        mask = apply_mask(color_mask, foreground)
        mask = apply_mask(mask, roi)
        #print('2')

        # display fg and color
        fg_resized =  resize_with_aspect_ratio(foreground, width=1280)
        color_resized = resize_with_aspect_ratio(color_mask, width=1280)
        cv2.imshow('foreground', fg_resized)
        cv2.imshow('color', color_resized)


        # apply mask to frame
        frame_processed = cv2.bitwise_and(frame,frame,mask=mask)
        frame_processed = erode_and_dilate(frame_processed)
        frame_processed = resize_with_aspect_ratio(frame_processed, width=1280)
        #print('3')

        interval_start = get_best_interval(mask, frame_width)
        interval_start_og = interval_start
        # displacement = position_history[-1] - interval_start
        # if direction != 0:
        #     # smaller than 200 for sanity
        #     if np.absolute(displacement) > interval_step and np.absolute(displacement) < 200:
        #         if np.sign(direction) == np.sign(displacement):
        #             interval_start = position_history[-1] + direction * interval_step
        #         else:
        #             if np.absolute(displacement) < 50:
        #                 interval_start = position_history[-1]
        # else:
        #     if np.absolute(displacement) > interval_step and np.absolute(displacement) < 200:
        #         interval_start = position_history[-1] + np.sign(displacement) * interval_step
        # interval_start = int(interval_start)
        # print(interval_start)
        # update history
        position_history = np.roll(position_history, -1)
        position_history[-1] = interval_start_og

        #get average direction
        total_displacement = 0
        for i in range(num_history - 1):
            total_displacement += position_history[i+1] - position_history[i]
        direction = -1 if total_displacement < 0 else 1 

        focus_start = min(interval_start - (forward_margin if direction < 0 else backward_margin), frame_width - interval_width - forward_margin - backward_margin)
        focus = frame[:, focus_start : (focus_start + interval_width + forward_margin + backward_margin)]
        if args['debug']:
            frame_w_rectangle = cv2.rectangle(frame, (focus_start, 0), (focus_start + interval_width + forward_margin + backward_margin, frame_height), (0,0,255), 3)
            #frame_w_rectangle = cv2.rectangle(frame, (interval_start, 0), (interval_start + interval_width , frame_height), (0,255,255), 3)
            frame_w_rectangle = resize_with_aspect_ratio(frame_w_rectangle, width=1280)

        if not skip:
            if args['save']:
                try:
                    writer.write(focus)
                except:
                    print('exception during writing')
            if args['save_imgs']:
                path = os.path.join(output_dir, str('{0:05d}'.format(frame_iter)) + '.jpg')
                print(path)
                cv2.imwrite(path, focus)
                with open(os.path.join(output_dir, args['video'].split('.')[0] + '.csv'), 'a') as file:
                    csv_writer = csv.writer(file, dialect='no_termination')
                    csv_writer.writerow([str(frame_iter) + '.jpg', str(focus_start), str(interval_width + forward_margin + backward_margin), str(frame_height)])
            if args['debug']:
                cv2.imshow('processed', frame_processed)
                cv2.imshow('og_frame', frame_w_rectangle)
                
                # wait for input
                c = cv2.waitKey(0)
                if 'q' == chr(c & 255):
                    #writer.release()
                    break
                elif 'n' == chr(c & 255):
                    pass
                elif 'f' == chr(c & 255):
                    skip(frame_gen, 50)
                    frame_iter += 50
        #print('5')
        frame_iter+=1
    if args['save']:
        print('releasing writer...')
        writer.release()

if __name__=='__main__':
    main()
