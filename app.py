import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull


from threading import Thread
import threading
import time
import pytz
import datetime
import traceback
import json
import uuid
from PIL import Image
from autocrop.autocrop import Cropper
from utils.pygifsicle import optimize



if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


# get configs

with open('config/config.yaml', 'r') as fd:
    config = yaml.safe_load(fd)

if torch.cuda.is_available():
    config['cpu']= False
    running_device = "cuda:0"
else:  #GPU is not available
    if not config['cpu']:
        config['cpu']= True
    running_device = None

# get the args and configs end 


# init global instances
def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        checkpoint_config = yaml.load(f)

    generator = OcclusionAwareGenerator(**checkpoint_config['model_params']['generator_params'],
                                        **checkpoint_config['model_params']['common_params'])

    if not cpu:
        torch.cuda.empty_cache()
        
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**checkpoint_config['model_params']['kp_detector_params'],
                             **checkpoint_config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

# check points
generator, kp_detector = load_checkpoints(config_path=config['checkpoint_config'], checkpoint_path=config['checkpoint'], cpu=config['cpu'])

# face cropper
g_cropper = Cropper(facecropper_label_path=config['facecropper_label_path'],facecropper_model_path=config['facecropper_model_path'],device=running_device)

# init global instances end


# functions
def del_file(file):
    if os.path.exists(file):
        print('delete {}',file)
        os.system("rm " + file)

def mp4togif(input_path,output_path):

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(output_path, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)

    writer.close()


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def worker(source_image_file, driving_video_file,gif_text=''):
    try:
        time_suffix = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d%H%M%S")
        one_uuid = str(uuid.uuid4())

        out_dir = config['out_dir']
        out_crop_file = out_dir+'/'+one_uuid+'.png'
        out_video_file = out_dir+'/'+one_uuid+'.mp4'
        out_video_file_with_text = out_dir+'/'+one_uuid+'_text'+'.mp4'
        out_gif_file = out_dir+'/'+ 'result_'+time_suffix + '.gif'


        # Get a Numpy array of the cropped image
        cropped_array = g_cropper.crop(source_image_file)

        # Save the cropped image with PIL if a face was detected:
        if cropped_array is 1:  # no need to crop
            out_crop_file = source_image_file
        elif cropped_array is 0: # no face in src image
            return None
        else: # crop face
            cropped_image = Image.fromarray(cropped_array)
            cropped_image.save(out_crop_file)
            

        source_image = imageio.imread(out_crop_file)
        reader = imageio.get_reader(driving_video_file)

        fps = reader.get_meta_data()['fps']
    
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        

        if config['find_best_frame'] or config['best_frame'] is not 0 :
            i = 0
            if config['best_frame']is not 0:
                i = config['best_frame']
            else: 
                i = find_best_frame(source_image, driving_video, cpu=config['cpu'])

            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=config['relative'], adapt_movement_scale=config['adapt_scale'], cpu=config['cpu'])
            predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=config['relative'], adapt_movement_scale=config['adapt_scale'], cpu=config['cpu'])
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=config['relative'], adapt_movement_scale=config['adapt_scale'], cpu=config['cpu'])

        imageio.mimsave(out_video_file, [img_as_ubyte(frame) for frame in predictions], fps=fps)



        if gif_text != '':
            os.system("ffmpeg -i '{}' -vf drawtext='fontcolor=white:fontsize=50:fontfile={}:line_spacing=7:text={}:x=(w-text_w)/2:y=(h-text_h)-10'  -y '{}'".format(
                                os.path.abspath(out_video_file), os.path.abspath(config['font']),gif_text,
                                os.path.abspath(out_video_file_with_text)))
            mp4togif(out_video_file_with_text,out_gif_file)
        else:
            mp4togif(out_video_file,out_gif_file)

        optimize(out_gif_file,colors=512)
                

    except:
        traceback.print_exc()
    else:
        print("worker task sucess")
        return out_gif_file
    finally:
        del_file(out_video_file)
        if out_crop_file != source_image_file:
            del_file(out_crop_file)
        del_file(out_video_file_with_text)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--source_image", default='data/input/Monalisa.png', help="path to source image")
    parser.add_argument("--driving_video", default='data/imitator_video/smile.mp4', help="path to driving video")
    parser.add_argument("--text", help="text add int gif animation",default='')
    opt = parser.parse_args()

    out = worker(opt.source_image,opt.driving_video,opt.text)
    if out is not None:
        print('out file is: ' + out )

    