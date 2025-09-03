"""
Image processing utilities for engineering applications.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


class ImageProcessor:
    """
    Basic image processing utilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Basic image preprocessing.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(blurred)
        
        return equalized
    
    def resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions."""
        return cv2.resize(image, (width, height))
    
    def crop(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Crop image to specified region."""
        return image[y:y+height, x:x+width]
    
    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h))
    
    def flip(self, image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
        """Flip image horizontally or vertically."""
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        else:
            return image
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return cv2.convertScaleAbs(image, alpha=1.0, beta=factor)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def apply_filter(self, image: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply various filters to image."""
        if filter_type == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == "median":
            return cv2.medianBlur(image, 5)
        elif filter_type == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            return cv2.filter2D(image, -1, kernel)
        else:
            return image
    
    def extract_basic_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract basic image features."""
        features = {
            "shape": image.shape,
            "dtype": image.dtype,
            "mean": np.mean(image),
            "std": np.std(image),
            "min": np.min(image),
            "max": np.max(image),
            "histogram": np.histogram(image, bins=256)[0].tolist()
        }
        
        return features
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "processor_type": "basic",
            "available_filters": ["gaussian", "median", "bilateral", "sharpen"]
        }


class EngineeringImageProcessor(ImageProcessor):
    """
    Specialized image processor for engineering applications.
    """
    
    def __init__(self):
        super().__init__()
        self.engineering_filters = {
            "edge_enhancement": self._edge_enhancement_filter,
            "noise_reduction": self._noise_reduction_filter,
            "contrast_enhancement": self._contrast_enhancement_filter,
            "detail_enhancement": self._detail_enhancement_filter
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Engineering-specific image preprocessing.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Basic preprocessing
        processed = super().preprocess(image)
        
        # Apply engineering-specific enhancements
        processed = self._enhance_engineering_features(processed)
        
        return processed
    
    def _enhance_engineering_features(self, image: np.ndarray) -> np.ndarray:
        """Enhance engineering-specific features in the image."""
        # Apply edge enhancement
        enhanced = self._edge_enhancement_filter(image)
        
        # Apply noise reduction
        enhanced = self._noise_reduction_filter(enhanced)
        
        # Apply contrast enhancement
        enhanced = self._contrast_enhancement_filter(enhanced)
        
        return enhanced
    
    def _edge_enhancement_filter(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges in the image."""
        # Apply Laplacian filter for edge enhancement
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine with original image
        enhanced = cv2.addWeighted(image, 0.8, laplacian, 0.2, 0)
        
        return enhanced
    
    def _noise_reduction_filter(self, image: np.ndarray) -> np.ndarray:
        """Reduce noise in the image."""
        # Apply bilateral filter for noise reduction while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        return filtered
    
    def _contrast_enhancement_filter(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast in the image."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _detail_enhancement_filter(self, image: np.ndarray) -> np.ndarray:
        """Enhance details in the image."""
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
    
    def detect_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect lines in the image."""
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                detected_lines.append({
                    "start": (x1, y1),
                    "end": (x2, y2),
                    "length": length,
                    "angle": angle
                })
        
        return detected_lines
    
    def detect_circles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect circles in the image."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append({
                    "center": (x, y),
                    "radius": r,
                    "area": np.pi * r**2
                })
        
        return detected_circles
    
    def detect_contours(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect contours in the image."""
        # Apply threshold
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            detected_contours.append({
                "contour": contour,
                "area": area,
                "perimeter": perimeter,
                "bounding_rect": (x, y, w, h),
                "aspect_ratio": aspect_ratio,
                "solidity": solidity
            })
        
        return detected_contours
    
    def detect_corners(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect corners in the image."""
        # Apply corner detection using Harris corner detector
        corners = cv2.cornerHarris(image, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Find corner coordinates
        corner_coords = np.where(corners > 0.01 * corners.max())
        
        detected_corners = []
        for i in range(len(corner_coords[0])):
            y, x = corner_coords[0][i], corner_coords[1][i]
            strength = corners[y, x]
            
            detected_corners.append({
                "position": (x, y),
                "strength": strength
            })
        
        return detected_corners
    
    def extract_engineering_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract engineering-specific features from the image."""
        features = {
            "basic_features": self.extract_basic_features(image),
            "lines": self.detect_lines(image),
            "circles": self.detect_circles(image),
            "contours": self.detect_contours(image),
            "corners": self.detect_corners(image),
            "engineering_metrics": self._calculate_engineering_metrics(image)
        }
        
        return features
    
    def _calculate_engineering_metrics(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate engineering-specific metrics."""
        # Calculate image complexity
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Calculate texture features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        texture_features = self._calculate_texture_features(gray)
        
        # Calculate geometric features
        geometric_features = self._calculate_geometric_features(image)
        
        return {
            "edge_density": edge_density,
            "texture_features": texture_features,
            "geometric_features": geometric_features,
            "complexity_score": self._calculate_complexity_score(image)
        }
    
    def _calculate_texture_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate texture features using Local Binary Patterns."""
        # Calculate LBP (simplified version)
        lbp = self._calculate_lbp(image)
        
        # Calculate texture statistics
        texture_stats = {
            "mean": np.mean(lbp),
            "std": np.std(lbp),
            "entropy": self._calculate_entropy(lbp),
            "uniformity": self._calculate_uniformity(lbp)
        }
        
        return texture_stats
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        # Simplified LBP implementation
        lbp = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                binary_string = ""
                
                # Check 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zero values
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate image uniformity."""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        uniformity = np.sum(hist**2)
        return uniformity
    
    def _calculate_geometric_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate geometric features."""
        # Detect lines and circles
        lines = self.detect_lines(image)
        circles = self.detect_circles(image)
        contours = self.detect_contours(image)
        
        # Calculate geometric metrics
        total_line_length = sum(line["length"] for line in lines)
        total_circle_area = sum(circle["area"] for circle in circles)
        total_contour_area = sum(contour["area"] for contour in contours)
        
        return {
            "line_count": len(lines),
            "total_line_length": total_line_length,
            "circle_count": len(circles),
            "total_circle_area": total_circle_area,
            "contour_count": len(contours),
            "total_contour_area": total_contour_area,
            "geometric_complexity": len(lines) + len(circles) + len(contours)
        }
    
    def _calculate_complexity_score(self, image: np.ndarray) -> float:
        """Calculate overall image complexity score."""
        # Edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Geometric complexity
        lines = self.detect_lines(image)
        circles = self.detect_circles(image)
        contours = self.detect_contours(image)
        geometric_complexity = len(lines) + len(circles) + len(contours)
        
        # Normalize and combine
        complexity_score = (edge_density * 10 + geometric_complexity / 100) / 2
        return min(complexity_score, 1.0)
    
    def apply_engineering_filter(self, image: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply engineering-specific filters."""
        if filter_type in self.engineering_filters:
            return self.engineering_filters[filter_type](image)
        else:
            return super().apply_filter(image, filter_type)
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        base_status = super().get_status()
        base_status.update({
            "processor_type": "engineering",
            "engineering_filters": list(self.engineering_filters.keys()),
            "available_detections": ["lines", "circles", "contours", "corners"]
        })
        return base_status
