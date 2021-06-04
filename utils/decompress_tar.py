import os
import glob

data = "/data/zming/datasets/MIDV2020/MIDV_det_final"
tar_files = glob.glob(os.path.join(data, "*.tar"))
tar_files.sort()

for file in tar_files:
    filename = str.split(file, "/")[-1]
    filename = filename[:-4]
    if not os.path.isdir(os.path.join(data, filename)):
        os.mkdir(os.path.join(data,filename))
    cmd = "tar xvf %s -C %s"%(file, os.path.join(data,filename))
    os.system(cmd)

