import cv2
import numpy as np

class MorphologicalSegmenter:
    def __init__(self):
        print("Initializing Lightweight Morphological Segmenter...")

    def predict_without_edge_detection(self, image, boxes):
        masks = []
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(gray_image.shape[1], x_max), min(gray_image.shape[0], y_max)
            
            crop = gray_image[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                continue

            kernel_size = (15, 15)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
            

            _, thresh = cv2.threshold(blackhat, 70, 255, cv2.THRESH_BINARY)
            
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_crop_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
            
            full_mask = np.zeros_like(gray_image, dtype=bool)
            full_mask[y_min:y_max, x_min:x_max] = cleaned_crop_mask > 0
            
            masks.append(full_mask)
            
        return masks

    def predict(self, image, boxes):

        masks = []
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(gray_image.shape[1], x_max), min(gray_image.shape[0], y_max)
            
            crop = gray_image[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                continue


            kernel_size = (15, 15)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
            blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
            
            blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
            

            edges = cv2.Canny(blurred, 30, 100)
            
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned_crop_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
            
            cleaned_crop_mask = cv2.dilate(cleaned_crop_mask, close_kernel, iterations=5)
            
            full_mask = np.zeros_like(gray_image, dtype=bool)
            full_mask[y_min:y_max, x_min:x_max] = cleaned_crop_mask > 0
            
            masks.append(full_mask)
            
        return masks