import cv2
import numpy as np

class BaselinePipeline:
    def __init__(self, detector, segmenter):
        self.detector = detector
        self.segmenter = segmenter

    def run(self, image_path, image, text_prompt):
        boxes = self.detector.predict(image_path, text_prompt)
        # import pdb;pdb.set_trace()
        masks = self.segmenter.predict(image, boxes)
        return boxes, masks


class HybridPipeline:
    def __init__(self, detector, segmenter):
        self.detector = detector
        self.segmenter = segmenter

   
    def _get_crack_centroid(self, image_gray, box):
        x_min, y_min, x_max, y_max = map(int, box)
        
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image_gray.shape[1], x_max), min(image_gray.shape[0], y_max)
        
        crop = image_gray[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
        
        blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 30, 100)
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        dilated_mask = cv2.dilate(closed_edges, close_kernel, iterations=5)
        
        coords = cv2.findNonZero(dilated_mask)
        if coords is not None:
            cx = int(np.mean(coords[:, 0, 0])) + x_min
            cy = int(np.mean(coords[:, 0, 1])) + y_min
            return [cx, cy]
            
        return None

    def run(self, image_path, image_rgb, prompt):
        boxes = self.detector.predict(image_rgb, prompt)
        
        if len(boxes) == 0:
            return [], []

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        points = []
        
        for box in boxes:
            if prompt == "crack":
                pt = self._get_crack_centroid(image_gray, box)
                points.append(pt)
            else:
                points.append(None) 

        masks = self.segmenter.predict_with_points(image_rgb, boxes, point_coords=points)
        
        return boxes, masks


# from models.segmenter.yolo_segmenter import YOLOSegmenter

class YOLOCascadePipeline:
    def __init__(self, detector, segmenter):
        self.detector = detector
        self.segmenter = segmenter

    def run(self, image_path, image_rgb, prompt):
        boxes = self.detector.predict(image_rgb, prompt)
        
        if len(boxes) == 0:
            return [], []

        masks = self.segmenter.predict(image_rgb, boxes)
        
        return boxes, masks

class PreprocessedPipeline:
    def __init__(self, detector, segmenter):
        self.detector = detector
        self.segmenter = segmenter

    def _dilate_and_burn(self, image_rgb, boxes):
        modified_image = image_rgb.copy()
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(image_gray.shape[1], x_max), min(image_gray.shape[0], y_max)
            
            crop = image_gray[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
            blurred = cv2.GaussianBlur(blackhat, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_edges = cv2.dilate(edges, close_kernel, iterations=3)
            
            crop_rgb = modified_image[y_min:y_max, x_min:x_max]
            crop_rgb[dilated_edges > 0] = [0, 0, 0]
            modified_image[y_min:y_max, x_min:x_max] = crop_rgb
            
        return modified_image

    def run(self, image_path, image_rgb, prompt):
        boxes = self.detector.predict(image_rgb, prompt)
        
        if len(boxes) == 0:
            return [], []

        if prompt == "crack":
            image_to_segment = self._dilate_and_burn(image_rgb, boxes)
        else:
            image_to_segment = image_rgb

        masks = self.segmenter.predict(image_to_segment, boxes)
        
        return boxes, masks