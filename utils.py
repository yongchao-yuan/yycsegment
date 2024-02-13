import cv2
import numpy as np
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


def get_yolo8_class(idx=None):
    cls_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
                24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
                29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
                37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
                75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                79: 'toothbrush'}
    if idx is None:
        return cls_dict
    elif idx in cls_dict:
        return cls_dict[idx]
    elif idx == -1:
        return list(cls_dict.keys())
    elif idx == -2:
        return list(cls_dict.values())
    else:
        return None


def load_local_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    image = np.array(Image.open(BytesIO(bytes_data)))
    return image


def save_mask_pic(inp_img, mask_lst, box_lst, background=None):
    """
    func: 根据掩码分割图片并填充背景
    para:
    img - numpy, 原图片
    mask_lst - 掩码列表(轮廓点是img坐标)
    box_lst - 矩形框列表(元素是list或tuple)
    background - None:透明，else: 填充的背景色
    """
    img = inp_img[:, :, :3]  # 只取前3个通道，png格式的有4个通道

    # make whole pic mask_template
    b_mask = np.zeros(img.shape[:2], np.uint8)

    # Create contour mask (点坐标的np数组[[22,122],[23,124],[33,112],...])
    for mask in mask_lst:
        contour = mask.astype(np.int32).reshape(-1, 1, 2)

        # 在mask_template上绘制mask
        _ = cv2.drawContours(
            b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    if background is None:
        # 独立对象带透明背景 (when saved as PNG)
        isolated = np.dstack([img, b_mask])
    else:
        # 转成BGR格式，和原图
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img)
        if background != (0, 0, 0):
            mask3ch = np.invert(mask3ch)
            b, g, r = cv2.split(mask3ch)
            b[:, :] = background[0]
            g[:, :] = background[1]
            r[:, :] = background[2]
            bgr = cv2.merge([b, g, r])
            back = cv2.bitwise_and(mask3ch, bgr)
            isolated = cv2.add(isolated, back)

    # 按照范围分割
    X = []
    Y = []
    for box in box_lst:
        x1, y1, x2, y2 = box.astype(np.int32)
        X.append(x1)
        X.append(x2)
        Y.append(y1)
        Y.append(y2)

    return isolated[min(Y):max(Y), min(X):max(X)]
