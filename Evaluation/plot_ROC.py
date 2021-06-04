import os
import numpy as np
import matplotlib.pyplot as plt

from read_bb_files import read_box
from box_iou import box_iou
from box_iou import label_pre_bboxes
from shutil import copyfile

folder = '/data/zming/logs/CBDAR/10000_calinorm'

file1_path = 'ROC_PR_evaluate_pcn_MIDV_v5_calibrate.txt'
file2_path = 'ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'
file3_path = 'ROC_PR_MIDV-500-outList_cascade_v6_calibrate.txt'




def main():
    fig_roc = plt.figure()
    fig_pr = plt.figure()
    files = [file1_path, file2_path, file3_path]
    strlegend= ['PCN','MTCNN','CascadeCNN']
    s=10000
    FP = np.zeros((3, s))
    TPR = np.zeros((3, s))
    PRES = np.zeros((3, s))
    for i in range(3):

        with open(os.path.join(folder, files[i]), 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines[1:]):
                line = str.split(line, ' ')

                FP[i][j] = int(line[1])
                TPR[i][j] = float(line[3])
                PRES[i][j] = float(line[4])


    ## plot ROC
    plt.figure(fig_roc.number)
    plt.plot(FP[0], TPR[0])
    plt.plot(FP[1], TPR[1])
    plt.plot(FP[2], TPR[2])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='--', linewidth=1)
    plt.legend(strlegend, loc='upper right')
    #plt.show()


    ## plot PR
    plt.figure(fig_pr.number)
    plt.plot(TPR[0], PRES[0])
    plt.plot(TPR[1], PRES[1])
    plt.plot(TPR[2], PRES[2])
    plt.xlabel('True Positive Rate')
    plt.ylabel('Precision')
    plt.grid(linestyle='--', linewidth=1)
    plt.legend(strlegend, loc='upper right')
    #plt.show()


    fig_roc.savefig(os.path.join(folder, 'ROC.png' ))

    fig_pr.savefig(os.path.join(folder, 'P-R.png'))




if __name__== '__main__':
    main()

