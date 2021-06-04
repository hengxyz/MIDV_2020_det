import os
import numpy as np
import matplotlib.pyplot as plt

from Evaluation.read_bb_files import read_box_MIDV2020_gt
from Evaluation.read_bb_files import read_box_MIDV2020_pred
from Evaluation.box_iou import box_iou
from Evaluation.box_iou import label_pre_bboxes
from shutil import copyfile

# folder = '/data/zming/datasets/CBDAR/'
folder_gt = '/data/zming/datasets/MIDV2020/MIDV_det_final/photo_bbox/annotations/'
folder_pred = '/data/zming/datasets/MIDV2020/photo_det/images/'

#file_path = 'evaluate_pcn_MIDV_v5_calibrate.txt'
#file_path = 'evaluate_mtcnn_MIDV_v5_calibrate.txt'
file_path = 'MIDV-500-outList_cascade_v6_calibrate.txt'

result='/data/zming/logs/CBDAR/'

def calcu_roc(gt_bboxes, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images):
    scale_roc = 10000
    TP = np.zeros(scale_roc)
    FP = np.zeros(scale_roc)
    PRECI = np.zeros(scale_roc)
    TPR = np.zeros(scale_roc)
    P = np.zeros(scale_roc)
    tp_images = []
    fp_images = []

    nrof_gt = 0
    for gt_bb in gt_bboxes:
        nrof_gt += len(gt_bb)
    GT = nrof_gt

    confi_scale = np.arange(0,1,1.0/scale_roc)
    confi_scale = confi_scale[::-1]

    # ## sort the pre_bboxes_iou according to the confidence
    # idx_sort = np.argsort(np.array(pre_bboxes_confidence))
    # pre_bboxes_confidence_sort = pre_bboxes_confidence(idx_sort)
    # pre_bboxes_label_sort = np.array(pre_bboxes_label)(idx_sort)
    # nrof_pre_bboxes = len(pre_bboxes_label_sort)


    for i, confi in enumerate(confi_scale):
        predict_bb_confi = np.greater(pre_bboxes_confidence, confi)
        nrof_predict_bb_confi = np.sum(predict_bb_confi)

        predict_bb_label = np.equal(pre_bboxes_label, 1)
        predict_bb = np.logical_and(predict_bb_confi, predict_bb_label)

        idx = range(len(predict_bb_label))
        idx_predic_bb = np.array(idx)[predict_bb]

        gt_recall_bboxes = np.array(gt_recall_bboxes_idx)[idx_predic_bb]
        gt_recall_bboxes_set = set (tuple(x) for x in gt_recall_bboxes)
        tp = len(gt_recall_bboxes_set)

        ########### Find the tp images and fp images #########################
        pre_iou_images_confi = np.array(pre_iou_images)[predict_bb_confi]
        pre_iou_images_almost_tp = np.array(pre_iou_images)[predict_bb]
        gt_recall_bboxes_set_idx = []
        for x in gt_recall_bboxes_set:
            notfind = True
            i = 0
            while(notfind):
                xx = gt_recall_bboxes_idx[i]
                if (np.array(x) == xx).all():
                    gt_recall_bboxes_set_idx.append(i)
                    notfind = False
                i += 1
        tp_images = np.array(pre_iou_images)[gt_recall_bboxes_set_idx]
        fp_images = np.array(pre_iou_images)[predict_bb_confi]
        fp_images = [x for x in fp_images if not x in tp_images]
        tp_images.sort()
        fp_images.sort()

        save_tp_images_path = os.path.join(result, file_path[:-4]+'_tp_images_%f'%confi)
        if not os.path.exists(save_tp_images_path):
            os.mkdir(save_tp_images_path)
        for img in tp_images:
            img = img.replace('/','_')
            copyfile(os.path.join(folder, file_path[:-4], img), os.path.join(save_tp_images_path, img))

        save_fp_images_path = os.path.join(result, file_path[:-4]+'_fp_images_%f'%confi)
        if not os.path.exists(save_fp_images_path):
            os.mkdir(save_fp_images_path)
        for img in fp_images:
            img = img.replace('/','_')
            copyfile(os.path.join(folder, file_path[:-4], img), os.path.join(save_fp_images_path, img))
        ##############################################################################

        p = nrof_predict_bb_confi
        fp = p - tp
        if p == 0:
            pr = 1.0
        else:
            pr = float(tp)/p
        tpr = float(tp)/GT

        TP[i] = tp
        FP[i] = fp
        PRECI[i] = pr
        TPR[i] = tpr
        P[i] = p






    return TP, FP, TPR, PRECI, P, tp_images, fp_images


def main():

    #gt_bb_principal, gt_bb_all, pre_bb, gt_bb_images, pre_bb_images =read_box(os.path.join(folder, file_path))
    gt_bb_principal =read_box_MIDV2020_gt(folder_gt)
    pred_bb_principal, pre_bb_images =read_box_MIDV2020_pred(folder_pred)
    #pre_bboxes_iou, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images = box_iou(gt_bb_all, pre_bb, pre_bb_images)
    pre_bboxes_iou, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images = box_iou(gt_bb_principal, pred_bb_principal, pre_bb_images, 0.5)

    # nrof_gt_bb = len(gt_bb_all)
    # TP, FP, TPR, PRECI, P, _, _ = calcu_roc(gt_bb_all, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images)
    nrof_gt_bb = len(gt_bb_principal)
    TP, FP, TPR, PRECI, P, _, _ = calcu_roc(gt_bb_principal, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images)
    with open(os.path.join(result, 'ROC_PR_%s.txt'%file_path[:-4]), 'w') as f:
        f.write('TP FP P TPR PRECI\n')
        for i in range(len(TP)):
            f.write('%d %d %d %f %f\n'%(TP[i],FP[i], P[i], TPR[i],PRECI[i]))

    ## plot ROC
    fig = plt.figure()
    plt.plot(FP, TPR)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate/Recall')
    plt.legend(loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(result, 'ROC_%s.png'%file_path[:-4]))

    ## plot PR
    fig = plt.figure()
    plt.plot(TPR, PRECI)
    plt.xlabel('True Positive Rate/Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(result, 'P-R_%s.png'%file_path[:-4]))











if __name__== '__main__':
    main()

