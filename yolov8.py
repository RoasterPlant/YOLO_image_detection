import torch as t
import random
import colorsys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from utils import manual_nms # função manual de NMS e IoU que devem implementar
from ultralytics import YOLO
from yolo import generate_colors, read_classes, preprocess_image

def executar_predicao(image_file, model, class_names, iou_threshold, score_threshold):
    image, image_data = preprocess_image(image_file, (416, 416)) 
    r = model(image_file)[0]
    r.save()
    resultados = r.boxes
    all_boxes = resultados.xyxy[:, [1, 0, 3, 2]]
    all_scores = resultados.conf
    all_classes = resultados.cls

    mask = all_scores >= score_threshold
    boxes = all_boxes[mask]
    scores = all_scores[mask]
    classes = all_classes[mask]

    #keep = torchvision.ops.nms(boxes, scores, iou_threshold) # Isso é função pronta, não use isso no seu código, a menos que queira comparar com a sua implementação apenas
    # Aplica o nosso NMS Manual
    keep = manual_nms(boxes, scores, classes, iou_threshold) # Aqui entra a função manual de NMS que devem implementar

    # Limita ao número máximo de caixas desejado (ex: top 10 detecções)
    max_boxes = 10
    keep = keep[:max_boxes]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    # Desenho
    colors = generate_colors(class_names)
    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(classes.cpu().numpy()))):
        predicted_class = class_names[int(c)]
        box = boxes[i].cpu().numpy()
        score = scores[i].cpu().item()

        label = f'{predicted_class} {score:.2f}'
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if left >= right or top >= bottom: continue

        text_origin = np.array([left, top - label_size[1]]) if top - label_size[1] >= 0 else np.array([left, top + 1])

        for j in range(thickness):
            if left+j >= right-j or top+j >= bottom-j: break
            draw.rectangle([left+j, top+j, right-j, bottom-j], outline=colors[int(c)])

        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[int(c)])
        draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        del draw

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    class_names = read_classes("data/coco.names") # Certifique-se de que o arquivo tenha 80 linhas
    iou_threshold = 0.5
    score_threshold = 0.5
    image_file = 'images/food.jpg'
    model = YOLO('yolov8x.pt')
    executar_predicao(image_file, model, class_names, iou_threshold, score_threshold)