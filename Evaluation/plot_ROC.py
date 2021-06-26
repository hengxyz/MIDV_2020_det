import os
import numpy as np
import matplotlib.pyplot as plt

from read_bb_files import read_box
from box_iou import box_iou
from box_iou import label_pre_bboxes
from shutil import copyfile

folder = '/data/zming/datasets/MIDV2020/Evluation_face_detection'
iou = 0.5
file1_path = '%s/clips_evaluation/iou_%0.1f/ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'%(folder, iou)
file2_path = '%s/Scan_rotated_evaluation/iou_%0.1f/ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'%(folder, iou)
file3_path = '%s/Scan_upright_evaluation/iou_%0.1f/ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'%(folder, iou)
file4_path = '%s/Photo_evaluation/iou_%0.1f/ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'%(folder, iou)
file5_path = '%s/Templates_evaluation/iou_%0.1f/ROC_PR_evaluate_mtcnn_MIDV_v5_calibrate.txt'%(folder, iou)




def main():
    fig_roc = plt.figure()
    fig_pr = plt.figure()
    files = [file1_path, file2_path, file3_path, file4_path, file5_path]
    strlegend= ['Clips','Scan_rotated','Scan_upright','Photo','Templates',]
    s=1000
    TP = np.zeros((len(files), s))
    FP = np.zeros((len(files), s))
    P = np.zeros((len(files), s))
    TPR = np.zeros((len(files), s))
    PRES = np.zeros((len(files), s))
    for i in range(len(files)):

        with open(os.path.join(files[i]), 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines[2:]):
                line = str.split(line, ' ')

                TP[i][j] = int(line[0])
                FP[i][j] = int(line[1])
                P[i][j] = int(line[2])
                TPR[i][j] = float(line[3])
                PRES[i][j] = float(line[4])


    ## plot ROC
    plt.figure(fig_roc.number)
    plt.plot(FP[0], TPR[0])
    plt.plot(FP[1], TPR[1])
    plt.plot(FP[2], TPR[2])
    plt.plot(FP[3], TPR[3])
    plt.plot(FP[4], TPR[4])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='--', linewidth=1)
    plt.legend(strlegend, loc='bottom right')
    #plt.show()


    ## plot PR
    plt.figure(fig_pr.number)
    plt.plot(TPR[0], PRES[0])
    plt.plot(TPR[1], PRES[1])
    plt.plot(TPR[2], PRES[2])
    plt.plot(TPR[3], PRES[3])
    plt.plot(TPR[4], PRES[4])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(linestyle='--', linewidth=1)
    plt.legend(strlegend, loc='bottom right')
    #plt.show()


    fig_roc.savefig(os.path.join(folder, 'ROC_%0.1f.png'%iou ))
    fig_roc.savefig(os.path.join(folder, 'ROC_%0.1f.svg'%iou ))

    fig_pr.savefig(os.path.join(folder, 'P-R_%0.1f.png'%iou))
    fig_pr.savefig(os.path.join(folder, 'P-R_%0.1f.svg'%iou))




if __name__== '__main__':
    main()

