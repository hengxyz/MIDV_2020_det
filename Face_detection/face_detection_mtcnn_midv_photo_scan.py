import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os
import glob
import time
import argparse
from MTCNN import MTCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'





scale = 0.25


parser = argparse.ArgumentParser(description='path_vids')
parser.add_argument('-p', '--path', type=str)
args = parser.parse_args()
path_vids = args.path
type_dir = str.split(path_vids, '/')[-1]
det_dir = type_dir+'_det'
path_det = path_vids.replace(type_dir,det_dir)


if not os.path.isdir(path_det):
    os.mkdir(path_det)
if not os.path.isdir(os.path.join(path_det, "images")):
    os.mkdir(os.path.join(path_det, "images"))
path_det = os.path.join(path_det, "images")
mtcnn = MTCNN('./models')



#images = glob.glob(os.path.join(path_vids, "images", '*.jpg '))
images = glob.glob(os.path.join(path_vids, "images", '*.tif '))
images.sort()
bbox_np = np.zeros(4)
num_images = len(images)

for k, image_path in enumerate(images):
    t0 = time.time()
    image_name = str.split(image_path, '/')[-1]
    image_name = image_name[:-4]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = [image.shape[0], image.shape[1]]
    image_rescale = cv2.resize(image, None, fx=scale, fy=scale)
    image_shape_rescale = [image_rescale.shape[0], image_rescale.shape[1]]
    bboxes, landmarks = mtcnn.detect_face(image_rescale)
    if len(bboxes) == 0:
        left, top, right, bottom = 0, 0, 0, 0
        confidence = 0.0
        with open(os.path.join(path_det, '%s_bbox.txt' % image_name), 'wt') as  f:
            f.write("".join("%d %d %d %d %f"%(left, top, right, bottom, confidence)) + "\n")
    else:
        # if len(bboxes)>1:
        #     print('bboxes plus 1')
        with open(os.path.join(path_det, '%s_bbox.txt' % image_name), 'wt') as  f:
            for bbox in bboxes:
                left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                confidence = bbox[4]
                left, top, right, bottom = int(left/scale), int(top/scale), int(right/scale), int(bottom/scale)
                f.write("".join("%d %d %d %d %f"%(left, top, right, bottom, confidence)) + "\n")
                cv2.rectangle(image, (left, top), (right, bottom),(0, 255, 0), 2, 8, 0)
    cv2.imwrite('%s.jpg'%os.path.join(path_det,image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


    t1 = time.time()
    print("mtcnn fps: %f [%d/%d] " % ((1 / (t1 - t0)), k, num_images))