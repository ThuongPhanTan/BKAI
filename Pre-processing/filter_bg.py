import numpy as np
import cv2

img_path = '../train_split/valid/images/img_0223.jpg'
txt_path = '../train_split/valid/labels/img_0223.jpg.txt'
img = cv2.imread(img_path)

def filter_bg(img, pts):
  rect = cv2.boundingRect(pts)
  x,y,w,h = rect
  croped = img[y:y+h, x:x+w].copy()

  pts = pts - pts.min(axis=0)
  mask = np.zeros(croped.shape[:2], np.uint8)
  cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

  dst = cv2.bitwise_and(croped, croped, mask=mask)
  bg = np.ones_like(croped, np.uint8)*255

  cv2.bitwise_not(bg,bg, mask=mask)
  cropped = bg+dst
  return cropped

with open(txt_path, 'r', encoding='utf8') as f:
  instance = f.readlines()

  for idx, ins in enumerate(instance):
    point = ins.split(',',8)[:-1]
    point = [int(p) for p in point]
    point = np.array(point).reshape(-1, 2)

    image = filter_bg(img, point)
    cv2.imwrite(f'{idx}.jpg', image)