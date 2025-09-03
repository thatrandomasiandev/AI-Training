"""
Advanced Computer Vision module for engineering applications.
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from .image_processor import EngineeringImageProcessor
from .cad_analyzer import CADAnalyzer
from .object_detector import EngineeringObjectDetector
from .feature_extractor import EngineeringFeatureExtractor
from .visual_inspector import QualityInspector
from ...utils.config import Config


class VisionModule:
    """
    Advanced Computer Vision module for engineering applications.
    
    Provides comprehensive CV capabilities including:
    - Engineering image processing
    - CAD drawing analysis
    - Object detection and recognition
    - Feature extraction
    - Visual inspection and quality control
    - Dimensional analysis
    - Defect detection
    """
    
    def __init__(self, config: Config, device: str = "cpu"):
        """
        Initialize the Vision module.
        
        Args:
            config: Configuration object
            device: Device to use for computations
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.image_processor = EngineeringImageProcessor()
        self.cad_analyzer = CADAnalyzer()
        self.object_detector = EngineeringObjectDetector(device)
        self.feature_extractor = EngineeringFeatureExtractor()
        self.visual_inspector = QualityInspector()
        
        # Load models
        self._load_models()
        
        self.logger.info("Vision Module initialized")
    
    def _load_models(self):
        """Load computer vision models."""
        try:
            # Load pre-trained models for object detection
            self.object_detector.load_models()
            
            # Load feature extraction models
            self.feature_extractor.load_models()
            
            # Load visual inspection models
            self.visual_inspector.load_models()
            
            self.logger.info("Vision models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading vision models: {e}")
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an engineering image using multiple CV techniques.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing image: {image_path}")
        
        # Load and preprocess image
        image = self._load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Basic image processing
        processed_image = self.image_processor.preprocess(image)
        
        # Extract basic features
        basic_features = self.image_processor.extract_basic_features(processed_image)
        
        # Object detection
        objects = await self.object_detector.detect_objects(processed_image)
        
        # Feature extraction
        features = await self.feature_extractor.extract_features(processed_image)
        
        # Visual inspection
        inspection_results = await self.visual_inspector.inspect_image(processed_image)
        
        # CAD analysis (if applicable)
        cad_analysis = self.cad_analyzer.analyze_drawing(processed_image)
        
        # Engineering-specific analysis
        engineering_analysis = self._analyze_engineering_content(processed_image, objects, features)
        
        return {
            "image_path": image_path,
            "image_shape": image.shape,
            "basic_features": basic_features,
            "objects": objects,
            "features": features,
            "inspection_results": inspection_results,
            "cad_analysis": cad_analysis,
            "engineering_analysis": engineering_analysis,
            "confidence": self._calculate_confidence(objects, features, inspection_results)
        }
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _analyze_engineering_content(self, image: np.ndarray, objects: List[Dict], features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engineering-specific content in the image."""
        analysis = {
            "has_technical_drawings": self._detect_technical_drawings(image),
            "has_measurements": self._detect_measurements(image),
            "has_dimensions": self._detect_dimensions(image),
            "has_annotations": self._detect_annotations(image),
            "has_symmetry": self._detect_symmetry(image),
            "has_geometric_shapes": self._detect_geometric_shapes(image),
            "engineering_objects": self._identify_engineering_objects(objects),
            "technical_complexity": self._assess_technical_complexity(image, objects, features)
        }
        
        return analysis
    
    def _detect_technical_drawings(self, image: np.ndarray) -> bool:
        """Detect if image contains technical drawings."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines (common in technical drawings)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        # If many lines detected, likely a technical drawing
        return lines is not None and len(lines) > 20
    
    def _detect_measurements(self, image: np.ndarray) -> bool:
        """Detect if image contains measurements."""
        # Look for measurement symbols and text
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OCR to detect text (simplified approach)
        # In practice, you'd use a proper OCR library like Tesseract
        # For now, detect patterns that look like measurements
        
        # Detect arrows and dimension lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
        
        return lines is not None and len(lines) > 10
    
    def _detect_dimensions(self, image: np.ndarray) -> bool:
        """Detect if image contains dimensional information."""
        # Look for dimension lines, arrows, and text
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect parallel lines (common in dimensioning)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return False
        
        # Check for parallel lines
        parallel_count = 0
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                
                # Calculate angle difference
                angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])
                angle_diff = abs(angle1 - angle2)
                
                if angle_diff < 0.1 or angle_diff > np.pi - 0.1:  # Nearly parallel
                    parallel_count += 1
        
        return parallel_count > 5
    
    def _detect_annotations(self, image: np.ndarray) -> bool:
        """Detect if image contains annotations."""
        # Look for text and symbols
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect contours that might be text
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small rectangular contours (likely text)
        text_contours = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Text-like contours are typically rectangular with specific aspect ratios
            if 0.2 < aspect_ratio < 5.0 and 10 < w < 200 and 5 < h < 50:
                text_contours += 1
        
        return text_contours > 5
    
    def _detect_symmetry(self, image: np.ndarray) -> bool:
        """Detect if image contains symmetric elements."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for horizontal symmetry
        h, w = gray.shape
        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)
        
        # Resize to match if needed
        if top_half.shape != bottom_half.shape:
            bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
        
        # Calculate similarity
        similarity = cv2.matchTemplate(top_half, bottom_half, cv2.TM_CCOEFF_NORMED)[0][0]
        
        return similarity > 0.7
    
    def _detect_geometric_shapes(self, image: np.ndarray) -> bool:
        """Detect if image contains geometric shapes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count geometric shapes
        geometric_shapes = 0
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a geometric shape
            if len(approx) >= 3:  # Triangle or more
                geometric_shapes += 1
        
        return geometric_shapes > 5
    
    def _identify_engineering_objects(self, objects: List[Dict]) -> List[str]:
        """Identify engineering objects from detected objects."""
        engineering_objects = []
        
        for obj in objects:
            class_name = obj.get("class", "").lower()
            
            # Map common object classes to engineering objects
            if "screw" in class_name or "bolt" in class_name:
                engineering_objects.append("fastener")
            elif "gear" in class_name:
                engineering_objects.append("gear")
            elif "bearing" in class_name:
                engineering_objects.append("bearing")
            elif "spring" in class_name:
                engineering_objects.append("spring")
            elif "valve" in class_name:
                engineering_objects.append("valve")
            elif "pump" in class_name:
                engineering_objects.append("pump")
            elif "motor" in class_name:
                engineering_objects.append("motor")
            elif "pipe" in class_name or "tube" in class_name:
                engineering_objects.append("pipe")
            elif "plate" in class_name or "sheet" in class_name:
                engineering_objects.append("plate")
            elif "beam" in class_name or "bar" in class_name:
                engineering_objects.append("beam")
        
        return list(set(engineering_objects))
    
    def _assess_technical_complexity(self, image: np.ndarray, objects: List[Dict], features: Dict[str, Any]) -> str:
        """Assess technical complexity of the image."""
        complexity_score = 0
        
        # Object complexity
        complexity_score += len(objects) * 0.1
        
        # Feature complexity
        if "edges" in features:
            complexity_score += len(features["edges"]) * 0.01
        
        if "corners" in features:
            complexity_score += len(features["corners"]) * 0.02
        
        # Image complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        complexity_score += np.sum(edges > 0) / (image.shape[0] * image.shape[1]) * 10
        
        if complexity_score > 5:
            return "high"
        elif complexity_score > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, objects: List[Dict], features: Dict[str, Any], inspection_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        confidence_scores = []
        
        # Object detection confidence
        if objects:
            obj_confidences = [obj.get("confidence", 0.0) for obj in objects]
            confidence_scores.append(np.mean(obj_confidences))
        
        # Feature extraction confidence
        if "confidence" in features:
            confidence_scores.append(features["confidence"])
        
        # Inspection confidence
        if "confidence" in inspection_results:
            confidence_scores.append(inspection_results["confidence"])
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    async def analyze_engineering_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple engineering images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Analysis results for all images
        """
        self.logger.info(f"Analyzing {len(image_paths)} engineering images")
        
        results = []
        for image_path in image_paths:
            try:
                analysis = await self.analyze_image(image_path)
                results.append({
                    "image": image_path,
                    "analysis": analysis,
                    "status": "success"
                })
            except Exception as e:
                self.logger.error(f"Error analyzing image {image_path}: {e}")
                results.append({
                    "image": image_path,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "images_analyzed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
    
    def extract_engineering_features(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract engineering-specific features from analysis."""
        features = {
            "technical_drawing": analysis.get("engineering_analysis", {}).get("has_technical_drawings", False),
            "measurements": analysis.get("engineering_analysis", {}).get("has_measurements", False),
            "dimensions": analysis.get("engineering_analysis", {}).get("has_dimensions", False),
            "annotations": analysis.get("engineering_analysis", {}).get("has_annotations", False),
            "symmetry": analysis.get("engineering_analysis", {}).get("has_symmetry", False),
            "geometric_shapes": analysis.get("engineering_analysis", {}).get("has_geometric_shapes", False),
            "engineering_objects": analysis.get("engineering_analysis", {}).get("engineering_objects", []),
            "technical_complexity": analysis.get("engineering_analysis", {}).get("technical_complexity", "low"),
            "object_count": len(analysis.get("objects", [])),
            "feature_count": len(analysis.get("features", {})),
            "inspection_passed": analysis.get("inspection_results", {}).get("passed", False)
        }
        
        return features
    
    def detect_defects(self, image_path: str) -> Dict[str, Any]:
        """
        Detect defects in engineering images.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Defect detection results
        """
        self.logger.info(f"Detecting defects in image: {image_path}")
        
        # Load image
        image = self._load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Use visual inspector for defect detection
        defects = self.visual_inspector.detect_defects(image)
        
        return {
            "image_path": image_path,
            "defects_detected": len(defects),
            "defects": defects,
            "defect_types": list(set([d.get("type", "unknown") for d in defects])),
            "severity": self._assess_defect_severity(defects)
        }
    
    def _assess_defect_severity(self, defects: List[Dict[str, Any]]) -> str:
        """Assess overall defect severity."""
        if not defects:
            return "none"
        
        # Calculate severity based on defect types and sizes
        severity_scores = []
        for defect in defects:
            defect_type = defect.get("type", "unknown")
            size = defect.get("size", 0)
            
            # Assign severity scores based on type
            if defect_type in ["crack", "fracture", "break"]:
                severity_scores.append(3.0)
            elif defect_type in ["scratch", "dent", "deformation"]:
                severity_scores.append(2.0)
            elif defect_type in ["stain", "discoloration", "corrosion"]:
                severity_scores.append(1.5)
            else:
                severity_scores.append(1.0)
            
            # Adjust for size
            if size > 100:  # Large defect
                severity_scores[-1] *= 1.5
            elif size > 50:  # Medium defect
                severity_scores[-1] *= 1.2
        
        avg_severity = np.mean(severity_scores)
        
        if avg_severity >= 2.5:
            return "high"
        elif avg_severity >= 1.5:
            return "medium"
        else:
            return "low"
    
    def measure_dimensions(self, image_path: str) -> Dict[str, Any]:
        """
        Measure dimensions in engineering images.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dimension measurement results
        """
        self.logger.info(f"Measuring dimensions in image: {image_path}")
        
        # Load image
        image = self._load_image(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Use CAD analyzer for dimension measurement
        dimensions = self.cad_analyzer.measure_dimensions(image)
        
        return {
            "image_path": image_path,
            "dimensions_measured": len(dimensions),
            "dimensions": dimensions,
            "measurement_confidence": self._calculate_measurement_confidence(dimensions)
        }
    
    def _calculate_measurement_confidence(self, dimensions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in dimension measurements."""
        if not dimensions:
            return 0.0
        
        confidences = [dim.get("confidence", 0.0) for dim in dimensions]
        return np.mean(confidences)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Vision module."""
        return {
            "device": self.device,
            "components": {
                "image_processor": self.image_processor.get_status(),
                "cad_analyzer": self.cad_analyzer.get_status(),
                "object_detector": self.object_detector.get_status(),
                "feature_extractor": self.feature_extractor.get_status(),
                "visual_inspector": self.visual_inspector.get_status()
            }
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.object_detector, 'cleanup'):
            self.object_detector.cleanup()
        if hasattr(self.feature_extractor, 'cleanup'):
            self.feature_extractor.cleanup()
        if hasattr(self.visual_inspector, 'cleanup'):
            self.visual_inspector.cleanup()
        
        self.logger.info("Vision Module cleanup complete")
