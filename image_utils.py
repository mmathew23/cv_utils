import cv2


def draw_bbox(img, xyxy, color=(255, 0, 0), thickness=2, label=None, probability=None):
    x1, y1, x2, y2 = [int(val) for val in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label or probability:
        label_txt = f'{label} ' if label else ''
        probability_txt = f'{probability*100:.0f}%' if probability else ''
        txt = label_txt + probability_txt
        cv2.putText(img, txt, [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
    return img
