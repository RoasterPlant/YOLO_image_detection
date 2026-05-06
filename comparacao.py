import json


def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0

    return inter_area / union


def calcular_iou_media(ground_truth, predicoes):
    """
    Para cada objeto real, procura a melhor predição da mesma classe.
    Se não houver predição correspondente, IoU = 0.
    """

    ious = []
    predicoes_usadas = set()

    for gt in ground_truth:
        melhor_iou = 0
        melhor_pred = None

        for j, pred in enumerate(predicoes):
            if j in predicoes_usadas:
                continue

            if gt["classe"] != pred["classe"]:
                continue

            iou = calcular_iou(gt["box"], pred["box"])

            if iou > melhor_iou:
                melhor_iou = iou
                melhor_pred = j

        ious.append(melhor_iou)

        if melhor_pred is not None:
            predicoes_usadas.add(melhor_pred)

    iou_media = sum(ious) / len(ious) if len(ious) > 0 else 0

    return iou_media, ious


def calcular_melhoria(iou_v3, iou_v8):
    """
    Calcula a melhoria percentual do melhor modelo em relação ao pior.
    """

    if iou_v3 == iou_v8:
        return "Empate", 0.0

    melhor_modelo = "YOLOv3" if iou_v3 > iou_v8 else "YOLOv8"
    melhor_iou = max(iou_v3, iou_v8)
    pior_iou = min(iou_v3, iou_v8)

    if pior_iou == 0:
        melhoria_percentual = float("inf")
    else:
        melhoria_percentual = ((melhor_iou - pior_iou) / pior_iou) * 100

    return melhor_modelo, melhoria_percentual


# ==============================
# LEITURA DOS ARQUIVOS
# ==============================

with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

with open("resultadosv3/predicoes_yolov3.json", "r") as f:
    predicoes_v3 = json.load(f)

with open("resultadosv8/predicoes_yolov8.json", "r") as f:
    predicoes_v8 = json.load(f)


# ==============================
# CÁLCULO DA IOU MÉDIA
# ==============================

iou_media_v3, ious_v3 = calcular_iou_media(
    ground_truth,
    predicoes_v3
)

iou_media_v8, ious_v8 = calcular_iou_media(
    ground_truth,
    predicoes_v8
)


# ==============================
# RESULTADOS INDIVIDUAIS
# ==============================

print("IoUs individuais - YOLOv3:")
for i, valor in enumerate(ious_v3):
    print(f"Objeto {i+1}: IoU = {valor:.4f}")

print(f"\nIoU média YOLOv3: {iou_media_v3:.4f}")


print("\n" + "=" * 40 + "\n")


print("IoUs individuais - YOLOv8:")
for i, valor in enumerate(ious_v8):
    print(f"Objeto {i+1}: IoU = {valor:.4f}")

print(f"\nIoU média YOLOv8: {iou_media_v8:.4f}")


# ==============================
# COMPARAÇÃO FINAL
# ==============================

melhor_modelo, melhoria_percentual = calcular_melhoria(
    iou_media_v3,
    iou_media_v8
)

print("\n" + "=" * 40)
print("COMPARAÇÃO FINAL")
print("=" * 40)

if melhor_modelo == "Empate":
    print("Resultado: empate entre YOLOv3 e YOLOv8.")
    print("Melhoria percentual: 0.00%")
else:
    melhor_iou = max(iou_media_v3, iou_media_v8)

    print(f"Melhor modelo: {melhor_modelo}")
    print(f"Melhor IoU médio: {melhor_iou:.4f}")

    if melhoria_percentual == float("inf"):
        print("Melhoria percentual: infinita, pois o pior modelo teve IoU médio igual a zero.")
    else:
        print(f"Melhoria percentual: {melhoria_percentual:.2f}%")