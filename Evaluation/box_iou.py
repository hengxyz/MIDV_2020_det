import numpy as np

def bb_iou(a, b):
    a_x_tl = a[2]
    a_y_tl = a[3]
    a_x_br = a[0]
    a_y_br = a[1]

    b_x_tl = b[2]
    b_y_tl = b[3]
    b_x_br = b[0]
    b_y_br = b[1]

    # a_x_tl = a[0]-a[2]
    # a_y_tl = a[1]-a[3]
    # a_x_br = a[0]
    # a_y_br = a[1]
    #
    # b_x_tl = b[0]-b[2]
    # b_y_tl = b[1]-b[3]
    # b_x_br = b[0]
    # b_y_br = b[1]

    x1 = max(a_x_tl, b_x_tl)
    y1 = max(a_y_tl, b_y_tl)
    x2 = min(a_x_br, b_x_br)
    y2 = min(a_y_br, b_y_br)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # set invalid entries to 0 overlap
    if w <= 0 or h <= 0:
        o = 0
    else:
        inter = w * h
        aarea = (a_x_br-a_x_tl+1)*(a_y_br-a_y_tl+1)
        barea = (b_x_br-b_x_tl+1)*(b_y_br-b_y_tl+1)
        # intersection over union overlap
        o = inter/ (aarea + barea - inter)


    return o

def box_iou(gt_bboxes, pre_bboxes, pre_bb_images, thresh_iou=0.5):
    nrof_images = np.shape(gt_bboxes)[0]
    pre_bboxes_iou = []
    pre_bboxes_confidence = []
    pre_bboxes_label = []
    gt_recall_bboxes_idx = []
    pre_iou_images = []
    for i in range(nrof_images):
        gt_bb_per_image = gt_bboxes[i]
        pre_bb_per_image = pre_bboxes[i]
        nrof_gt_bboxes = len(gt_bb_per_image)
        nrof_pre_bboxes = len(pre_bb_per_image)
        for j in range(nrof_pre_bboxes):
            pre_bb = pre_bb_per_image[j]
            for k in range(nrof_gt_bboxes):
                gt_bb = gt_bb_per_image[k]
                pre_bb_iou=bb_iou(gt_bb, pre_bb)
                if pre_bb_iou > thresh_iou:
                    label = 1
                else:
                    label = 0
                pre_bboxes_iou.append(pre_bb_iou)
                #pre_bboxes_confidence.append(pre_bb[4])
                pre_bboxes_confidence.append(1.0)
                pre_bboxes_label.append(label)
                gt_recall_bboxes_idx.append([i,k])
                pre_iou_images.append(pre_bb_images[i])

    ### Normalisation the cofidence of the bounding boxes
    confi_max = max(pre_bboxes_confidence)
    confi_min = min(pre_bboxes_confidence)
    #pre_bboxes_confidence_norm = [(x-confi_min)/(confi_max-confi_min) for x in pre_bboxes_confidence]
    pre_bboxes_confidence_norm = [1.0 for x in pre_bboxes_confidence]

    return pre_bboxes_iou, pre_bboxes_confidence_norm, pre_bboxes_label, gt_recall_bboxes_idx, pre_iou_images

def label_pre_bboxes(pre_bboxes_iou, thresh_iou=0.5):
    labels_pre_bboxes = []
    for pre_bb_iou in pre_bboxes_iou:
        if pre_bb_iou[0]>thresh_iou:
            label = 1
        else:
            label = 0


        labels_pre_bboxes.append(label)

    return labels_pre_bboxes