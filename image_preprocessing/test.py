import cv2
import os
import numpy as np

input_dir = '/Users/gmansmacbook/Documents/DeepX_G/image_preprocessing/test'
output_dir = '/Users/gmansmacbook/Documents/DeepX_G/image_preprocessing/test'

def resize_image(img, width=None, height=None):
    """調整影像大小"""
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height))
    elif width is not None:
        r = width / img.shape[1]
        dim = (width, int(img.shape[0] * r))
        img = cv2.resize(img, dim)
    elif height is not None:
        r = height / img.shape[0]
        dim = (int(img.shape[1] * r), height)
        img = cv2.resize(img, dim)
    return img

def gray_scale(img):
    """轉換為灰度影像"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def remove_noise(img, kernel_size=(3, 3)):
    """去除噪聲"""
    blurred = cv2.GaussianBlur(img, kernel_size, 0)
    return blurred

def enhance_contrast(img, alpha=1.2, beta=20):
    """增強對比度"""
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return enhanced

def thresholding(img, thresh=127, maxval=255, type=cv2.THRESH_BINARY):
    """二值化"""
    ret, threshed = cv2.threshold(img, thresh, maxval, type)
    return threshed

def edge_detection(img, low_threshold=100, high_threshold=200):
    """邊緣檢測"""
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges


for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        
        # Create a new filename for the output image
        output_filename = os.path.splitext(filename)[0] + '_processed.png'
        output_path = os.path.join(output_dir, output_filename)
        
        # 讀取影像
        img = cv2.imread(input_path)

        # 調整影像大小
        img = resize_image(img, width=800)

        # 轉換為灰度影像
        gray = gray_scale(img)

        # 去除噪聲
        blurred = remove_noise(gray)

        # 增強對比度
        enhanced = enhance_contrast(blurred)

        # 二值化
        threshed = thresholding(enhanced)

        # 邊緣檢測
        edges = edge_detection(threshed)

        # 將處理結果輸出至指定路徑的PNG檔案中
        cv2.imwrite(os.path.join(output_dir, 'gray_scale_' + filename), gray)
        cv2.imwrite(os.path.join(output_dir, 'blurred_' + filename), blurred)
        cv2.imwrite(os.path.join(output_dir, 'enhanced_contrast_' + filename), enhanced)
        cv2.imwrite(os.path.join(output_dir, 'thresholded_' + filename), threshed)
        cv2.imwrite(os.path.join(output_dir, 'edge_detection_' + filename), edges)

