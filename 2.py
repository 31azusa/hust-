import cv2
import numpy as np
from skimage import filters
import imutils

# 读取图像
img = cv2.imread('pg/school.jpg')

# 预处理，将较长的边的分辨率调整到256
h, w = img.shape[:2]
if h > w:
    new_h = 256
    new_w = int(w * new_h / h)
else:
    new_w = 256
    new_h = int(h * new_w / w)
img = cv2.resize(img, (new_w, new_h))

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算图像的视觉显著性
edges = filters.sobel(gray)
saliency = np.abs(edges)

# 将显著性图像进行二值化处理
threshold = filters.threshold_otsu(saliency)
binary = saliency > threshold * 0.71  #方形头像0.71，圆形头像0.55

# 检测显著的目标
contours = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
scores = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    score = np.sum(saliency[y:y+h, x:x+w])
    scores.append((score, (x, y, w, h)))
scores.sort(reverse=True)
top_scores = scores[:4]

# 输出前三高分数的框
for i, (score, (x, y, w, h)) in enumerate(top_scores):
    # 计算正方形的边长，不超过原图最短边的长度
    side = min(w, h, new_h, new_w)
    # 计算正方形左上角的坐标
    x1 = x + (w - side) // 2
    y1 = y + (h - side) // 2
    # 计算正方形右下角的坐标
    x2 = x1 + side
    y2 = y1 + side
    # 绘制正方形框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{i+1}: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Saliency Map', binary.astype(np.float32))
cv2.waitKey(0)
cv2.destroyAllWindows()