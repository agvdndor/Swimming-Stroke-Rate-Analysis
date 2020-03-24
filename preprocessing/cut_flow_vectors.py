import cv2
import numpy as np
import argparse
import sys
from tqdm import tqdm
import os
from os import path as osp
import csv
from glob import glob

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='Path to the the directory containing stitched and stitched_optical_vectors')
    ap.add_argument('-d', '--debug', required=False, action='store_true',
                    help='Show debug views such as roi, foreground and step through video frame per frame')
    args = vars(ap.parse_args())
    return args

# dictionary with key = img_number and value = x_start, width
def get_image_cut_coordinates(filename):
    cut_coordinates = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            fields = line.split(',')
            cut_coordinates[fields[0].split('.')[0]] = [int(fields[1]), int(fields[2])]
    return cut_coordinates

def flo_to_numpy(flo_url):
    with open(flo_url, mode='r') as flo:
        tag = np.fromfile(flo, np.float32, count=1)[0]
        width = np.fromfile(flo, np.int32, count=1)[0]
        height = np.fromfile(flo, np.int32, count=1)[0]
        print('tag', tag, 'width', width, 'height', height)
        nbands = 2
        tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
        flow = np.resize(tmp, (int(height), int(width), int(nbands)))
    return flow


def main():
    # get arguments
    args = get_arguments()

    # coordinates files name
    print(args)
    print(args['root'])
    print(osp.join(args['root'], 'stitched', 'stitched.csv'))
    cut_coordinates_url = glob(osp.join(args['root'], 'stitched', 'stitched.csv'))[0]
    print(cut_coordinates_url)

    # get coordinates
    cut_coordinates = get_image_cut_coordinates(cut_coordinates_url)

    # get flo urls
    #flo_urls = glob(osp('stitched_optical_vectors', '*.flo'))
    if not os.path.exists(os.path.join(args['root'], 'optical_vectors')):
        os.mkdir(os.path.join(args['root'], 'optical_vectors'))

    for img_id in cut_coordinates.keys():
        cut_co = cut_coordinates[img_id]
        print(cut_co)
        # get flu url
        # add extra zero to img_id lol
        flo_url = osp.join(args['root'],'stitched_optical_vectors', str(img_id).zfill(6) + '.flo')
        try:
            flo_np = flo_to_numpy(flo_url)
        except: 
            print('flow files {} not found'.format(flo_url))
            continue
        
        # save numpy array as csv
        output_path = osp.join(args['root'], 'optical_vectors', str(img_id).zfill(5) + '.csv')
        np.savetxt(output_path, flo_np, delimiter=',')


if __name__=='__main__':
    main()