import torch
import numpy as np


def manual_nms(boxes, scores, classes, iou_threshold):
    keep = []
    box_queues = {item:[] for item in classes.tolist()} #Ideia: criar uma fila de caixas para cada classe detectada. 
    _, indices = torch.sort(scores, descending=True)
    sorted_classes = classes[indices]

    for i in range(sorted_classes.numel()):
        box_queues[sorted_classes[i].item()].append(indices[i].item())

    for queue in box_queues.values():
        queue = torch.Tensor(queue).int()
        while queue.numel() > 0:
            keep.append(queue[0].item())
            box_iou = iou(boxes[queue])
            queue = queue[box_iou < iou_threshold]

    return torch.Tensor(keep).int()


def iou(boxes):
    main_box = boxes[0]
    main_area = box_area(main_box)
    n = boxes.size(dim=0) 
    iou_areas = np.ones(n)

    for i in range(1, n):
        box = boxes[i]
        box_pair = torch.vstack((main_box, box))
        intersection_box = torch.Tensor([torch.amax(box_pair[:, 0], 0), torch.amin(box_pair[:, 1], 0), torch.amax(box_pair[:, 2], 0), torch.amin(box_pair[:, 3], 0)])
        intersection_area = box_area(intersection_box)
        iou_areas[i] = intersection_area/(main_area + box_area(box) - intersection_area)
    
    return torch.Tensor(iou_areas)

def box_area(box):
    return torch.mul((box[2] - box[0]), (box[3] - box[1])).item()