"""
CAD drawing analysis utilities for engineering applications.
"""

import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
import re
from scipy import ndimage
from skimage import measure, morphology
import matplotlib.pyplot as plt


class CADAnalyzer:
    """
    Basic CAD drawing analysis utilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.drawing_elements = {
            "lines": [],
            "circles": [],
            "arcs": [],
            "dimensions": [],
            "text": [],
            "symbols": []
        }
    
    def analyze_drawing(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze CAD drawing elements.
        
        Args:
            image: Input image
            
        Returns:
            Analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect drawing elements
        lines = self._detect_lines(gray)
        circles = self._detect_circles(gray)
        arcs = self._detect_arcs(gray)
        dimensions = self._detect_dimensions(gray)
        text = self._detect_text(gray)
        symbols = self._detect_symbols(gray)
        
        # Analyze drawing structure
        structure = self._analyze_drawing_structure(gray, lines, circles, arcs)
        
        return {
            "lines": lines,
            "circles": circles,
            "arcs": arcs,
            "dimensions": dimensions,
            "text": text,
            "symbols": symbols,
            "structure": structure,
            "drawing_type": self._classify_drawing_type(lines, circles, arcs, dimensions)
        }
    
    def _detect_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect lines in the drawing."""
        # Apply edge detection
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
                    "angle": angle,
                    "type": self._classify_line_type(angle, length)
                })
        
        return detected_lines
    
    def _detect_circles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect circles in the drawing."""
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
                    "diameter": 2 * r,
                    "area": np.pi * r**2,
                    "type": self._classify_circle_type(r)
                })
        
        return detected_circles
    
    def _detect_arcs(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect arcs in the drawing."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_arcs = []
        for contour in contours:
            # Check if contour is arc-like
            if self._is_arc(contour):
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Calculate arc properties
                arc_length = cv2.arcLength(contour, False)
                arc_angle = self._calculate_arc_angle(contour, (x, y), radius)
                
                detected_arcs.append({
                    "center": (int(x), int(y)),
                    "radius": radius,
                    "length": arc_length,
                    "angle": arc_angle,
                    "contour": contour
                })
        
        return detected_arcs
    
    def _detect_dimensions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect dimension lines and text."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        
        dimensions = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line looks like a dimension line
                if self._is_dimension_line(line[0], image):
                    # Look for dimension text near the line
                    text = self._extract_dimension_text(line[0], image)
                    
                    dimensions.append({
                        "line": (x1, y1, x2, y2),
                        "text": text,
                        "value": self._parse_dimension_value(text),
                        "unit": self._extract_dimension_unit(text)
                    })
        
        return dimensions
    
    def _detect_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text in the drawing."""
        # Apply threshold
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour looks like text
            if self._is_text_region(w, h, cv2.contourArea(contour)):
                text_regions.append({
                    "region": (x, y, w, h),
                    "area": cv2.contourArea(contour),
                    "aspect_ratio": w / h if h > 0 else 0
                })
        
        return text_regions
    
    def _detect_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect engineering symbols in the drawing."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        symbols = []
        for contour in contours:
            # Check if contour looks like a symbol
            if self._is_symbol(contour):
                x, y, w, h = cv2.boundingRect(contour)
                symbol_type = self._classify_symbol(contour)
                
                symbols.append({
                    "region": (x, y, w, h),
                    "type": symbol_type,
                    "contour": contour
                })
        
        return symbols
    
    def _classify_line_type(self, angle: float, length: float) -> str:
        """Classify line type based on angle and length."""
        if abs(angle) < 5 or abs(angle - 180) < 5:
            return "horizontal"
        elif abs(angle - 90) < 5 or abs(angle + 90) < 5:
            return "vertical"
        elif abs(angle - 45) < 5 or abs(angle + 45) < 5:
            return "diagonal"
        else:
            return "angled"
    
    def _classify_circle_type(self, radius: float) -> str:
        """Classify circle type based on radius."""
        if radius < 10:
            return "small"
        elif radius < 50:
            return "medium"
        else:
            return "large"
    
    def _is_arc(self, contour: np.ndarray) -> bool:
        """Check if contour is arc-like."""
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)
        
        # Check if it's arc-like based on area and perimeter
        if area < 100 or perimeter < 50:
            return False
        
        # Check circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return 0.3 < circularity < 0.9
    
    def _calculate_arc_angle(self, contour: np.ndarray, center: Tuple[float, float], radius: float) -> float:
        """Calculate arc angle."""
        # Find start and end points
        distances = [np.sqrt((p[0][0] - center[0])**2 + (p[0][1] - center[1])**2) for p in contour]
        start_idx = np.argmin(distances)
        end_idx = np.argmax(distances)
        
        # Calculate angles
        start_angle = np.arctan2(contour[start_idx][0][1] - center[1], contour[start_idx][0][0] - center[0])
        end_angle = np.arctan2(contour[end_idx][0][1] - center[1], contour[end_idx][0][0] - center[0])
        
        return abs(end_angle - start_angle) * 180 / np.pi
    
    def _is_dimension_line(self, line: Tuple[int, int, int, int], image: np.ndarray) -> bool:
        """Check if line is a dimension line."""
        x1, y1, x2, y2 = line
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Dimension lines are typically short and have arrows or text nearby
        return 20 < length < 200
    
    def _extract_dimension_text(self, line: Tuple[int, int, int, int], image: np.ndarray) -> str:
        """Extract dimension text near a line."""
        x1, y1, x2, y2 = line
        
        # Define search region around the line
        margin = 20
        x_min = max(0, min(x1, x2) - margin)
        x_max = min(image.shape[1], max(x1, x2) + margin)
        y_min = max(0, min(y1, y2) - margin)
        y_max = min(image.shape[0], max(y1, y2) + margin)
        
        # Extract region
        region = image[y_min:y_max, x_min:x_max]
        
        # Apply OCR (simplified - in practice, use Tesseract)
        # For now, return placeholder
        return "dimension_text"
    
    def _parse_dimension_value(self, text: str) -> Optional[float]:
        """Parse dimension value from text."""
        # Extract numerical value from text
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return float(numbers[0])
        return None
    
    def _extract_dimension_unit(self, text: str) -> str:
        """Extract dimension unit from text."""
        # Common units
        units = ['mm', 'cm', 'm', 'in', 'ft', '°', 'deg']
        for unit in units:
            if unit in text.lower():
                return unit
        return "unknown"
    
    def _is_text_region(self, width: int, height: int, area: float) -> bool:
        """Check if region looks like text."""
        aspect_ratio = width / height if height > 0 else 0
        
        # Text regions typically have specific aspect ratios and sizes
        return (0.2 < aspect_ratio < 5.0 and 
                10 < width < 200 and 
                5 < height < 50 and 
                area > 50)
    
    def _is_symbol(self, contour: np.ndarray) -> bool:
        """Check if contour looks like a symbol."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)
        
        # Symbols are typically small and have specific shapes
        return 50 < area < 1000 and perimeter > 20
    
    def _classify_symbol(self, contour: np.ndarray) -> str:
        """Classify symbol type."""
        # Calculate contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)
        
        # Approximate contour
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify based on number of vertices
        vertices = len(approx)
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            return "rectangle"
        elif vertices > 8:
            return "circle"
        else:
            return "polygon"
    
    def _analyze_drawing_structure(self, image: np.ndarray, lines: List[Dict], circles: List[Dict], arcs: List[Dict]) -> Dict[str, Any]:
        """Analyze overall drawing structure."""
        structure = {
            "total_elements": len(lines) + len(circles) + len(arcs),
            "line_count": len(lines),
            "circle_count": len(circles),
            "arc_count": len(arcs),
            "complexity": self._calculate_drawing_complexity(lines, circles, arcs),
            "symmetry": self._detect_symmetry(image),
            "layers": self._detect_layers(image)
        }
        
        return structure
    
    def _calculate_drawing_complexity(self, lines: List[Dict], circles: List[Dict], arcs: List[Dict]) -> str:
        """Calculate drawing complexity."""
        total_elements = len(lines) + len(circles) + len(arcs)
        
        if total_elements > 100:
            return "high"
        elif total_elements > 50:
            return "medium"
        else:
            return "low"
    
    def _detect_symmetry(self, image: np.ndarray) -> Dict[str, bool]:
        """Detect symmetry in the drawing."""
        h, w = image.shape[:2]
        
        # Check horizontal symmetry
        top_half = image[:h//2, :]
        bottom_half = cv2.flip(image[h//2:, :], 0)
        
        if top_half.shape != bottom_half.shape:
            bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
        
        h_similarity = cv2.matchTemplate(top_half, bottom_half, cv2.TM_CCOEFF_NORMED)[0][0]
        
        # Check vertical symmetry
        left_half = image[:, :w//2]
        right_half = cv2.flip(image[:, w//2:], 1)
        
        if left_half.shape != right_half.shape:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        
        v_similarity = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0][0]
        
        return {
            "horizontal": h_similarity > 0.7,
            "vertical": v_similarity > 0.7
        }
    
    def _detect_layers(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect different layers in the drawing."""
        # Apply threshold to separate different intensity levels
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        layers = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100:  # Filter small components
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                layers.append({
                    "id": i,
                    "area": area,
                    "bounding_box": (x, y, w, h),
                    "centroid": (int(centroids[i][0]), int(centroids[i][1]))
                })
        
        return layers
    
    def _classify_drawing_type(self, lines: List[Dict], circles: List[Dict], arcs: List[Dict], dimensions: List[Dict]) -> str:
        """Classify the type of drawing."""
        total_elements = len(lines) + len(circles) + len(arcs)
        dimension_count = len(dimensions)
        
        if dimension_count > 10:
            return "detailed_drawing"
        elif total_elements > 50:
            return "assembly_drawing"
        elif len(circles) > len(lines):
            return "circular_drawing"
        else:
            return "general_drawing"
    
    def measure_dimensions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Measure dimensions in the drawing."""
        # Analyze drawing
        analysis = self.analyze_drawing(image)
        
        # Extract dimension measurements
        measurements = []
        for dimension in analysis["dimensions"]:
            if dimension["value"] is not None:
                measurements.append({
                    "value": dimension["value"],
                    "unit": dimension["unit"],
                    "line": dimension["line"],
                    "text": dimension["text"],
                    "confidence": 0.8  # Placeholder confidence
                })
        
        return measurements
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "analyzer_type": "cad",
            "drawing_elements": list(self.drawing_elements.keys()),
            "available_analyses": ["lines", "circles", "arcs", "dimensions", "text", "symbols"]
        }


class DrawingAnalyzer(CADAnalyzer):
    """
    Specialized drawing analyzer for engineering drawings.
    """
    
    def __init__(self):
        super().__init__()
        self.engineering_symbols = {
            "welding": self._detect_welding_symbols,
            "surface_finish": self._detect_surface_finish_symbols,
            "tolerance": self._detect_tolerance_symbols,
            "datum": self._detect_datum_symbols
        }
    
    def analyze_engineering_drawing(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze engineering drawing with specialized features."""
        # Basic CAD analysis
        basic_analysis = self.analyze_drawing(image)
        
        # Engineering-specific analysis
        engineering_analysis = {
            "welding_symbols": self._detect_welding_symbols(image),
            "surface_finish": self._detect_surface_finish_symbols(image),
            "tolerances": self._detect_tolerance_symbols(image),
            "datums": self._detect_datum_symbols(image),
            "materials": self._detect_material_symbols(image),
            "processes": self._detect_process_symbols(image)
        }
        
        # Combine analyses
        combined_analysis = {**basic_analysis, "engineering": engineering_analysis}
        
        return combined_analysis
    
    def _detect_welding_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect welding symbols in the drawing."""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        welding_symbols = []
        for contour in contours:
            if self._is_welding_symbol(contour):
                x, y, w, h = cv2.boundingRect(contour)
                welding_symbols.append({
                    "type": "welding",
                    "region": (x, y, w, h),
                    "contour": contour
                })
        
        return welding_symbols
    
    def _detect_surface_finish_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect surface finish symbols."""
        # Similar implementation for surface finish symbols
        return []
    
    def _detect_tolerance_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect tolerance symbols."""
        # Similar implementation for tolerance symbols
        return []
    
    def _detect_datum_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect datum symbols."""
        # Similar implementation for datum symbols
        return []
    
    def _detect_material_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect material symbols."""
        # Similar implementation for material symbols
        return []
    
    def _detect_process_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect process symbols."""
        # Similar implementation for process symbols
        return []
    
    def _is_welding_symbol(self, contour: np.ndarray) -> bool:
        """Check if contour is a welding symbol."""
        # Implement welding symbol detection logic
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)
        
        # Welding symbols have specific characteristics
        return 100 < area < 1000 and perimeter > 30
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        base_status = super().get_status()
        base_status.update({
            "analyzer_type": "engineering_drawing",
            "engineering_symbols": list(self.engineering_symbols.keys())
        })
        return base_status
