import os
import numpy as np
import cv2

image_path = '/data/zming/datasets/midv-500'
cbdar_path = '/data/zming/datasets/CBDAR/'
evaluate_data = 'evaluate_pcn_MIDV_v5_calibrate.txt'
#evaluate_data = 'evaluate_mtcnn_MIDV_v5_calibrate.txt'
#evaluate_data = 'MIDV-500-outList_cascade_v6_calibrate.txt'



def main():
    num_images = 0
    num_no_ground_truth = 0
    num_no_detected_face = 0

    GT_bboxes = []
    GT_bboxes_principle = []
    Pre_bboxes = []

    images_folder = str.split(evaluate_data, '.')[0]
    if not os.path.exists(os.path.join(cbdar_path, images_folder)):
        os.mkdir(os.path.join(cbdar_path, images_folder))
    with open(os.path.join(cbdar_path, evaluate_data), 'r') as f:
        lines = f.readlines()
        for line in lines:
            num_images += 1
            line = str.split(line, ' ')
            img_file = line[0]
            ## read image
            img = cv2.imread(os.path.join(image_path, img_file))

            ## read the ground truth of bounding box
            gt_num = int(line[1])
            if not gt_num:
                print('===============>>>> No groud truth of bounding box in %s!'%img_file)
                num_no_ground_truth += 1
                continue
            gt_bboxes = np.zeros((gt_num, 4))

            ## read groud truth of bounding box
            for i in range(gt_num):
                i_start = 2+i*4
                i_end = 2+(i+1)*4
                gt_bboxes[i] = list(map(int, line[i_start:i_end]))

                ## draw the ground truth
                cv2.rectangle(img, (int(gt_bboxes[i][0]), int(gt_bboxes[i][1])),
                              (int(gt_bboxes[i][2]), int(gt_bboxes[i][3])), (255, 0, 0), 3)

            gt_idx_end = i_end



            ## read the predicted bounding box and the confidences
            pre_num = (len(line)-gt_idx_end)/5
            if pre_num == 0:
                print('%s No face has been detected!'%img_file)
                num_no_detected_face += 1
            else:
                pre_bboxes = np.zeros((pre_num, 5))
                for i in range(pre_num):
                    i_start = gt_idx_end + i * 5
                    i_end = gt_idx_end + (i + 1) * 5
                    pre_bboxes[i, 0:4] = list(map(int, line[i_start:i_end-1]))

                    ## draw the predicted boxes
                    shfit_pixel = 5
                    cv2.rectangle(img, (int(pre_bboxes[i][0]+shfit_pixel), int(pre_bboxes[i][1]+shfit_pixel)),
                                  (int(pre_bboxes[i][2]+shfit_pixel), int(pre_bboxes[i][3])+shfit_pixel), (0, 255, 0), 3, 8)
            img_new_file = img_file.replace('/','_')
            cv2.imwrite(os.path.join(cbdar_path,images_folder,img_new_file), img)


        print('Total %d images, %d No ground truth, %d No detected face!'%(num_images, num_no_ground_truth, num_no_detected_face))

    return GT_bboxes, GT_bboxes_principle, Pre_bboxes




if __name__ == '__main__':
    main()
