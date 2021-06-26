import os
import glob

#path = "/data/zming/datasets/MIDV2020/clips_det/images/"
#path = "/data/zming/datasets/MIDV2020/perfect_scan_det/images/"
#path = "/data/zming/datasets/MIDV2020/photo_det/images/"
path = "/data/zming/datasets/MIDV2020/shift_scan_det/images/"

folders = os.listdir(path)
folders.sort()

for folder in folders:
    print("---------%s"%folder)
    xx = str.split(folder, '_')
    new_folder = "%s_%02d_%02d"%(xx[0],int(xx[1]),int(xx[2]))
    os.rename(os.path.join(path,folder), os.path.join(path,new_folder))

# for folder in folders:
#     print("---------%s"%folder)
#     zz = str.split(folder, '.')
#     xx = str.split(zz[0], '_')
#     new_folder = "%s_%02d_%02d"%(xx[0],int(xx[1]),int(xx[2]))
#     os.rename(os.path.join(path,folder), os.path.join(path,new_folder+'.%s'%zz[1]))

