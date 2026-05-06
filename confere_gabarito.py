import cv2
import json

img = cv2.imread("images/paisagem.jpg")

with open("ground_truth.json", "r") as f:
    gt = json.load(f)

for item in gt:

    x1, y1, x2, y2 = item["box"]

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        item["classe"],
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1
    )

cv2.imshow("Ground Truth", img)

cv2.waitKey(0)
cv2.destroyAllWindows()