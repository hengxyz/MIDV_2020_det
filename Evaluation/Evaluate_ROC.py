import os
import numpy as np
import matplotlib.pyplot as plt

# from Evaluation.read_bb_files import read_box_MIDV2020_gt_templates
# from Evaluation.read_bb_files import read_box_MIDV2020_pred_templates
# from Evaluation.read_bb_files import read_box_MIDV2020_gt
# from Evaluation.read_bb_files import read_box_MIDV2020_pred
from read_bb_files import read_box_MIDV2020_gt_clips
from read_bb_files import read_box_MIDV2020_pred_clips
from box_iou import box_iou
# from Evaluation.box_iou import label_pre_bboxes
# from shutil import copyfile
# from scipy import interpolate
#type_dir = "templates_det"
#type_dir = "shift_scan_det"
#type_dir = "photo_det"
type_dir = "clips_det"
miss_annotation_file = "/data/zming/datasets/MIDV2020/MIDV_det_final/clips/miss_annotation_file.txt"
folder = '/data/zming/datasets/MIDV2020/%s/evaluation'%type_dir
folder_gt = '/data/zming/datasets/MIDV2020/MIDV_det_final/clips_bbox/annotations/'
#folder_pred = '/data/zming/datasets/MIDV2020/%s/images/'%type_dir
folder_pred = '/data/zming/datasets/MIDV2020/MIDV_det_final/%s/images/'%type_dir

#file_path = 'evaluate_pcn_MIDV_v5_calibrate.txt'
file_path = 'evaluate_mtcnn_MIDV_v5_calibrate.txt'
#file_path = 'MIDV-500-outList_cascade_v6_calibrate.txt'

#result='/data/zming/logs/CBDAR/'
result = '/data/zming/datasets/MIDV2020/%s/evaluation'%type_dir

IOU_THRESHOLD = 0.7
def calcu_roc(gt_bboxes, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images):
    scale_roc = 1000
    #scale_roc = 100
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
        print("Evaluation: %f [%d/%d]"%(confi, i, scale_roc))
        predict_bb_confi = np.greater(pre_bboxes_confidence, confi)
        nrof_predict_bb_confi = np.sum(predict_bb_confi) ## positive bbox, i.e. confidence > threshold

        predict_bb_label = np.equal(pre_bboxes_label, 1)
        predict_bb = np.logical_and(predict_bb_confi, predict_bb_label)

        idx = range(len(predict_bb_label))
        idx_predic_bb = np.array(idx)[predict_bb]

        gt_recall_bboxes = np.array(gt_recall_bboxes_idx)[idx_predic_bb]
        gt_recall_bboxes_set = set (tuple(x) for x in gt_recall_bboxes)
        tp = len(gt_recall_bboxes_set) ## true positive bbox

        # ########### Find the tp images and fp images #########################
        # pre_iou_images_confi = np.array(pre_iou_images)[predict_bb_confi]
        # pre_iou_images_almost_tp = np.array(pre_iou_images)[predict_bb]
        # gt_recall_bboxes_set_idx = []
        # for x in gt_recall_bboxes_set:
        #     notfind = True
        #     j = 0
        #     while(notfind):
        #         xx = gt_recall_bboxes_idx[i]
        #         if (np.array(x) == xx).all():
        #             gt_recall_bboxes_set_idx.append(i)
        #             notfind = False
        #         j += 1
        # tp_images = np.array(pre_iou_images)[gt_recall_bboxes_set_idx]
        # fp_images = np.array(pre_iou_images)[predict_bb_confi]
        # fp_images = [x for x in fp_images if not x in tp_images]
        # tp_images.sort()
        # fp_images.sort()

        # save_tp_images_path = os.path.join(result, 'tp_images')
        # if not os.path.exists(save_tp_images_path):
        #     os.mkdir(save_tp_images_path)
        # for img in tp_images:
        #     # img = img.replace('/','_')
        #     # copyfile(os.path.join(folder, file_path[:-4], img), os.path.join(save_tp_images_path, img))
        #     img_name = str.split(img, '/')[-1]
        #     copyfile(img, os.path.join(save_tp_images_path, img_name))
        #
        # save_fp_images_path = os.path.join(result, 'fp_images')
        # if not os.path.exists(save_fp_images_path):
        #     os.mkdir(save_fp_images_path)
        # for img in fp_images:
        #     # img = img.replace('/','_')
        #     #copyfile(os.path.join(folder, file_path[:-4], img), os.path.join(save_fp_images_path, img))
        #     img_name = str.split(img, '/')[-1]
        #     copyfile(img, os.path.join(save_fp_images_path, img_name))

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

    AP = np.mean(PRECI)




    return TP, FP, TPR, PRECI, P, AP, tp_images, fp_images


def main():
    if not os.path.isdir(result):
        os.mkdir(result)
    result_path = os.path.join(result, "iou_%.1f"%IOU_THRESHOLD)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    gt_bb_principal = read_box_MIDV2020_gt_clips(folder_gt)
    pred_bb_principal, pre_bb_images =read_box_MIDV2020_pred_clips(folder_pred, miss_annotation_file)
    #pred_bb_principal, pre_bb_images =read_box_MIDV2020_pred_templates(folder_pred)
    #pre_bboxes_iou, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images = box_iou(gt_bb_all, pre_bb, pre_bb_images)
    pre_bboxes_iou, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images = box_iou(gt_bb_principal, pred_bb_principal, pre_bb_images, IOU_THRESHOLD)

    # nrof_gt_bb = len(gt_bb_all)
    # TP, FP, TPR, PRECI, P, _, _ = calcu_roc(gt_bb_all, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images)
    nrof_gt_bb = len(gt_bb_principal)
    TP, FP, TPR, PRECI, P, AP, _, _ = calcu_roc(gt_bb_principal, pre_bboxes_confidence, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images)
    with open(os.path.join(result_path, 'ROC_PR_%s.txt'%file_path[:-4]), 'w') as f:
        f.write('AP@IoU: %f@%f\n'%(AP, IOU_THRESHOLD))
        f.write('TP FP P TPR PRECI\n')
        for i in range(len(TP)):
            f.write('%d %d %d %f %f \n'%(TP[i],FP[i], P[i], TPR[i],PRECI[i]))

    ## plot ROC
    # f = interpolate.interp1d(FP, TPR)
    # FP_new = np.linspace(0,max(FP),1001)
    # TPR_new = f(FP_new)
    fig = plt.figure()
    plt.plot(FP, TPR)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive Rate/Recall')
    plt.legend(loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(result_path, 'ROC_%s.png'%file_path[:-4]))

    ## plot PR
    for i in range(len(PRECI)):
        PRECI[i] = max(PRECI[i:])
    # f = interpolate.interp1d(TPR, PRECI)
    # TPR_new = np.linspace(0,1,1001)
    # PRECI_new = f(TPR_new)
    fig = plt.figure()
    plt.plot(TPR, PRECI)
    plt.xlabel('True Positive Rate/Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    #plt.show()
    fig.savefig(os.path.join(result_path, 'P-R_%s.png'%file_path[:-4]))











if __name__== '__main__':
    main()

