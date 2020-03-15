import cv2
import numpy as np
import sys
import os
import csv
import argparse
import glob
from tqdm import tqdm

input_dir = r"C:\Users\Vande\data_preprocessed"

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True,
                    help='Path to the videos directory')
    args = vars(ap.parse_args())
    return args

def main():
    # get arguments
    args = get_arguments()

    video_dir = args['video']
    #frames = glob.glob('*.jpg')
    csv_list = glob.glob(os.path.join(video_dir,'stitched','*.csv'))

    if len(csv_list) != 1:
        print('too many csv files')
        print(csv_list)
        
    else:
        print(csv_list)
        csv_file = csv_list[0]
        

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        lines = list(csv_reader)
    
    print(lines[0])
    first_entry = lines[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(video_dir, 'recreated', args['video'] + '_recreated.avi') , fourcc, 25.0, (int(first_entry[2]),int(first_entry[3])))

    for line in tqdm(lines):
        # read image
        img = cv2.imread(os.path.join(video_dir,'images', line[0]))   
        writer.write(img)
    
    writer.release()

if __name__ == "__main__":
    main()
