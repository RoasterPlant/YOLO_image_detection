import cv2
import json

image_path = "images/paisagem.jpg"

img = cv2.imread(image_path)

ground_truth = []

while True:

    # Digite a classe
    classe = input("Classe do objeto (ou ENTER para terminar): ")

    if classe == "":
        break

    print("Selecione o objeto e aperte ENTER")

    box = cv2.selectROI(
        "Imagem",
        img,
        fromCenter=False,
        showCrosshair=True
    )

    x, y, w, h = box

    if w == 0 or h == 0:
        print("Caixa inválida.")
        continue

    ground_truth.append({
        "classe": classe,
        "box": [
            int(x),
            int(y),
            int(x + w),
            int(y + h)
        ]
    })

    print("Objeto salvo.")

cv2.destroyAllWindows()

# Salvar JSON
with open("ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=4)

print("Ground truth salvo.")