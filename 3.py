import cv2
import numpy as np
from skimage import filters
import imutils
from sklearn.cluster import KMeans

# 读取图像
img = cv2.imread('pg/lunwen3.png')

# 预处理，将较长的边的分辨率调整到512
h, w = img.shape[:2]
if h > w:
    new_h = 512
    new_w = int(w * new_h / h)
else:
    new_w = 512
    new_h = int(h * new_w / w)
img = cv2.resize(img, (new_w, new_h))

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算图像的视觉显著性
# 使用高斯金字塔进行多尺度计算
saliency = np.zeros_like(gray, dtype=np.float32)
for scale in range(1, 3):
    # 缩小图像
    scaled_img = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
    # 计算边缘
    edges = filters.sobel(scaled_img)
    # 加权计算显著性分数
    saliency += cv2.resize(edges, (gray.shape[1], gray.shape[0]))

# 将显著性图像进行二值化处理
threshold = filters.threshold_otsu(saliency)
binary = saliency > threshold * 0.5 # 方形头像0.71，圆形头像0.55

# 检测显著的目标
contours = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
scores = []
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w >= 40 and h >= 40:  # 添加限制条件，排除非常小的目标框
        score = np.sum(saliency[y:y+h, x:x+w])
        scores.append((score, (x, y, w, h)))
scores.sort(reverse=True)
top_scores = scores[:1]  # 修改为最高分数的目标框

# 获取显著性区域的质心/中心

centroids = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:  # 排除面积为零的区域
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))


# 生成正方形框
for i, (score, (x, y, w, h)) in enumerate(top_scores):
    # 寻找最接近的质心
    ccx = x + h/2
    ccy = y + w/2
    min_distance = float('inf')
    closest_centroid = None
    for centroid in centroids:
        cx, cy = centroid
        distance = np.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_centroid = centroid

    if closest_centroid is not None:
        cx, cy = closest_centroid
        # 计算正方形边长
        side = min(w, h)
        # 计算正方形左上角的坐标
        x1 = cx - side // 2
        y1 = cy - side // 2
        # 计算正方形右下角的坐标
        x2 = x1 + side
        y2 = y1 + side
        # 绘制正方形框
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(img, f'{i + 1}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 获取待处理部分的交集

if x1<0 :
    x1=0
if y1<0 :
    y1=0
if x2>new_w :
    x2=new_w
if y1>new_h :
    y2=new_h

roi = img[y1:y2, x1:x2]

# 检查roi图像是否为空
if roi.size != 0:
    # 转换为灰度图像
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 计算ROI的视觉显著性
    saliency_roi = np.zeros_like(gray_roi, dtype=np.float32)
    for scale in range(1, 3):
        # 缩小图像
        scaled_roi = cv2.resize(gray_roi, (gray_roi.shape[1] // scale, gray_roi.shape[0] // scale))
        # 计算边缘
        edges_roi = filters.sobel(scaled_roi)
        # 加权计算显著性分数
        saliency_roi += cv2.resize(edges_roi, (gray_roi.shape[1], gray_roi.shape[0]))

    # 将显著性图像进行二值化处理
    threshold_roi = filters.threshold_otsu(saliency_roi)
    binary_roi = saliency_roi > threshold_roi * 0.9  # 方形头像0.71，圆形头像0.55

    # 检测显著的目标
    contours_roi = cv2.findContours(binary_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_roi = imutils.grab_contours(contours_roi)
    scores_roi = []
    for contour_roi in contours_roi:
        (x_roi, y_roi, w_roi, h_roi) = cv2.boundingRect(contour_roi)
        if w_roi >= 10 and h_roi >= 10:  # 添加限制条件，排除非常小的目标框
            score_roi = np.sum(saliency_roi[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi])
            scores_roi.append((score_roi, (x_roi, y_roi, w_roi, h_roi)))
    scores_roi.sort(reverse=True)
    top_scores_roi = scores_roi[:1]  # 修改为最高分数的目标框

    # 生成正方形框
    for i_roi, (score_roi, (x_roi, y_roi, w_roi, h_roi)) in enumerate(top_scores_roi):
        # 计算正方形左上角的坐标
        w_max = max(w_roi,h_roi)
        x1_roi = x_roi + x1
        y1_roi = y_roi + y1
        # 计算正方形右下角的坐标
        x2_roi = x1_roi + w_max
        y2_roi = y1_roi + w_max
        # 绘制正方形框
        if x1_roi < 0:
            x1_roi = 0
        if y1_roi < 0:
            y1_roi = 0
        if x2_roi > new_w:
            x2_roi = new_w
        if y2_roi > new_h:
            y2_roi = new_h

        # 计算矩形的宽度和高度
        width = abs(x2_roi - x1_roi)
        height = abs(y2_roi - y1_roi)
        # 计算矩形的中心点
        center_x = (x1_roi + x2_roi) // 2
        center_y = (y1_roi + y2_roi) // 2
        # 计算内接圆的半径
        radius = min(width, height) // 2

        # 绘制内接圆
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)
        cv2.rectangle(img, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)
        cv2.putText(img, f'{i + 1}: {score_roi:.2f}', (x1_roi, y1_roi - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Saliency Map', binary.astype(np.float32))
# 提取内接圆所包围的部分图像
roi_img = img[y1_roi:y2_roi, x1_roi:x2_roi]
# 创建一个新窗口来显示内接圆所包围的部分图像
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
# 在新窗口中显示内接圆所包围的部分图像
cv2.imshow("ROI", roi_img)
cv2.waitKey(0)
cv2.destroyAllWindows()