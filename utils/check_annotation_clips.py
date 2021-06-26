import os
import glob
folder_images = "/data/zming/datasets/MIDV2020/MIDV_det_final/clips/images/"
folder_bbox = "/data/zming/datasets/MIDV2020/MIDV_det_final/clips_bbox/annotations/"
miss_annotation_file = "/data/zming/datasets/MIDV2020/MIDV_det_final/clips/miss_annotation_file.txt"

subs = os.listdir(folder_images)
subs.sort()

with open(miss_annotation_file, 'w') as f:
    for sub in subs:
        img_folders = os.listdir(os.path.join(folder_images, sub))
        img_folders.sort()
        for folder in img_folders:
            imgs = glob.glob(os.path.join(folder_images, sub, folder, '*.jpg'))
            imgs.sort()
            len_imgs = len(imgs)
            bboxes = glob.glob(os.path.join(folder_bbox, sub, folder+'_bbox', '*.txt'))
            bboxes.sort()
            len_bboxes = len(bboxes)
            if len_imgs != len_bboxes:
                bbox_names = [str.split(bbox,'/')[-1] for bbox in bboxes]
                for img in imgs:
                    img_name = str.split(img, '/')[-1]
                    img_name = img_name[:-4]
                    bbox = img_name+'_bbox.txt'
                    if bbox not in bbox_names:
                        print("%s not annotated"%img)
                        f.write("%s\n"%img)
