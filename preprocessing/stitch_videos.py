import os 
import glob
import subprocess
from tqdm import tqdm

data_dir = r"C:\Users\Vande\data_og"
styles = ["freestyle", "butterfly", "breaststroke"]

# dict with indices of relevant videos for every style
indices = {}
indices['freestyle'] = [3,4,5,0]
indices['butterfly'] = [0,3,4,5,6]
indices['breaststroke'] = [0,3,4,5,6]
indices['backstroke'] = [0,3,4,5,6]

motion_compensation = True


def main():
    for style in styles:
        style_dir = os.path.join(data_dir, style)
        print("starting style: ", style)

        # get all samples (numbered 1 to ...)
        samples = glob.glob(os.path.join(style_dir, '*'))
        print("found ", str(len(samples))," samples..." )

        for sample in tqdm(samples):
            sample_dir =os.path.join(style_dir, sample)

            # if the stitched video already exists
            if not os.path.exists(os.path.join(sample_dir, 'stitched')):
                os.mkdir(os.path.join(sample_dir, 'stitched'))
            stitched_vid = glob.glob(os.path.join(sample_dir, "stitched", "*"))
            if len(stitched_vid) > 0:
                print("stitched video already exists for sample ", str(sample))       
                continue

            # no stitched video present yet
            vid_dir = os.path.join(sample_dir, "videos")
            vids = glob.glob(os.path.join(vid_dir, "*.avi"))
            print(vids)
            input_string = ""
            for ind in indices[style]:
                try:
                    input_string += "-i "+ vids[ind] + " "
                except:
                    continue

            print("string of inputs: " + input_string)

            stitched_dir = os.path.join(sample_dir, "stitched")
            output_path = os.path.join(stitched_dir, "stitched.avi")
            #ffmpeg call
            try:
                if len(indices[style]) == 5:
                    subprocess.call("ffmpeg " + input_string + "-filter_complex \"[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]hstack[v];[v][4:v]hstack[q]\" -map \"[q]\" -shortest -vcodec libx264 "+ output_path)                
                else:
                    subprocess.call("ffmpeg " + input_string + "-filter_complex \"[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]hstack[v]\" -map \"[v]\" -shortest -vcodec libx264 "+ output_path)
            except:
                print('Fail for video: ' + sample_dir)

            if motion_compensation:
                subprocess.call('python motion_compensation.py --video ' + output_path + ' --save_imgs')


        # ffmpeg -i Freestyle_-_---_-__7.avi -i Freestyle_-_---_-__8.avi -i Freestyle_-_---_-__9.avi -i Freestyle_-_---_-__10.avi -filter_complex "[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]hstack[v]" -map "[v]" -shortest -vcodec libx264 stitched.mp4

if __name__ == "__main__":
    main()

