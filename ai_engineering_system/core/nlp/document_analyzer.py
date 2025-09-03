"""
Document analysis utilities for engineering applications.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import PyPDF2
import docx
import pandas as pd
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import fitz  # PyMuPDF
import cv2
import pytesseract
from PIL import Image
import io


class DocumentAnalyzer:
    """
    Basic document analysis utilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.pdf', '.docx', '.txt', '.html', '.csv', '.xlsx']
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from various document formats.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        elif file_extension == '.html':
            return self._extract_from_html(file_path)
        elif file_extension == '.csv':
            return self._extract_from_csv(file_path)
        elif file_extension == '.xlsx':
            return self._extract_from_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            # Try PyMuPDF first (better for complex PDFs)
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e2:
                self.logger.error(f"Both PDF extraction methods failed: {e2}")
                return ""
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error extracting from DOCX: {e}")
            return ""
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                self.logger.error(f"Error reading TXT file: {e}")
                return ""
    
    def _extract_from_html(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()
        except Exception as e:
            self.logger.error(f"Error extracting from HTML: {e}")
            return ""
    
    def _extract_from_csv(self, file_path: Path) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            self.logger.error(f"Error extracting from CSV: {e}")
            return ""
    
    def _extract_from_xlsx(self, file_path: Path) -> str:
        """Extract text from XLSX file."""
        try:
            df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            self.logger.error(f"Error extracting from XLSX: {e}")
            return ""
    
    def extract_from_url(self, url: str) -> str:
        """Extract text from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Check if it's HTML
            if 'text/html' in response.headers.get('content-type', ''):
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup.get_text()
            else:
                return response.text
        except Exception as e:
            self.logger.error(f"Error extracting from URL: {e}")
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "supported_formats": self.supported_formats,
            "analyzer_type": "basic"
        }


class TechnicalDocumentAnalyzer(DocumentAnalyzer):
    """
    Specialized document analyzer for technical/engineering documents.
    """
    
    def __init__(self):
        super().__init__()
        self.technical_sections = [
            'abstract', 'introduction', 'methodology', 'results', 'discussion',
            'conclusion', 'references', 'appendix', 'specifications', 'requirements',
            'design', 'analysis', 'testing', 'validation', 'implementation'
        ]
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of a technical document."""
        structure = {
            "sections": self._identify_sections(text),
            "tables": self._extract_tables(text),
            "figures": self._extract_figure_references(text),
            "equations": self._extract_equations(text),
            "references": self._extract_references(text),
            "metadata": self._extract_metadata(text)
        }
        
        return structure
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections."""
        sections = []
        lines = text.split('\n')
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks like a section header
            if self._is_section_header(line):
                # Save previous section
                if current_section:
                    sections.append({
                        "title": current_section,
                        "content": '\n'.join(current_content),
                        "line_number": i
                    })
                
                # Start new section
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content),
                "line_number": len(lines)
            })
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header."""
        # Common patterns for section headers
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # All caps
            r'^[A-Z][a-z]+\s+[A-Z]',  # Title case
            r'^[A-Z][a-z]+:',  # Title with colon
        ]
        
        import re
        for pattern in patterns:
            if re.match(pattern, line):
                return True
        
        # Check against known technical sections
        line_lower = line.lower()
        for section in self.technical_sections:
            if section in line_lower:
                return True
        
        return False
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract table-like structures from text."""
        tables = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line looks like table data
            if self._is_table_row(line):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and len(current_table) > 1:
                    tables.append({
                        "content": current_table,
                        "line_number": i - len(current_table)
                    })
                in_table = False
                current_table = []
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like table data."""
        # Look for patterns with multiple columns (tabs, multiple spaces, or pipes)
        if '\t' in line:
            return True
        
        if '|' in line:
            return True
        
        # Check for multiple numbers or measurements
        import re
        measurements = re.findall(r'\d+(?:\.\d+)?\s*(?:mm|cm|m|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)', line)
        if len(measurements) >= 2:
            return True
        
        return False
    
    def _extract_figure_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract figure references from text."""
        import re
        
        figure_patterns = [
            r'Figure\s+(\d+(?:\.\d+)?)',
            r'Fig\.\s*(\d+(?:\.\d+)?)',
            r'Figure\s+(\d+(?:\.\d+)?)\s*[:\-]\s*(.+)',
            r'Fig\.\s*(\d+(?:\.\d+)?)\s*[:\-]\s*(.+)'
        ]
        
        figures = []
        for pattern in figure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                figures.append({
                    "number": match.group(1),
                    "caption": match.group(2) if len(match.groups()) > 1 else "",
                    "position": match.start()
                })
        
        return figures
    
    def _extract_equations(self, text: str) -> List[Dict[str, Any]]:
        """Extract equations from text."""
        import re
        
        equation_patterns = [
            r'Equation\s+(\d+(?:\.\d+)?)',
            r'Eq\.\s*(\d+(?:\.\d+)?)',
            r'\((\d+(?:\.\d+)?)\)\s*[A-Za-z]\s*[=<>≤≥]\s*[A-Za-z0-9\+\-\*\/\(\)\^]+',
            r'[A-Za-z]\s*[=<>≤≥]\s*[A-Za-z0-9\+\-\*\/\(\)\^]+'
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                equations.append({
                    "equation": match.group(0),
                    "number": match.group(1) if len(match.groups()) > 0 else None,
                    "position": match.start()
                })
        
        return equations
    
    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references from text."""
        import re
        
        reference_patterns = [
            r'\[(\d+(?:,\s*\d+)*)\]',  # [1, 2, 3]
            r'\(([A-Za-z]+\s+\d{4})\)',  # (Smith 2023)
            r'([A-Za-z]+\s+et\s+al\.\s+\d{4})',  # Smith et al. 2023
            r'([A-Za-z]+\s+\d{4})'  # Smith 2023
        ]
        
        references = []
        for pattern in reference_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                references.append({
                    "reference": match.group(0),
                    "position": match.start()
                })
        
        return references
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract document metadata."""
        metadata = {
            "word_count": len(text.split()),
            "character_count": len(text),
            "line_count": len(text.split('\n')),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "has_abstract": 'abstract' in text.lower(),
            "has_references": bool(self._extract_references(text)),
            "has_figures": bool(self._extract_figure_references(text)),
            "has_equations": bool(self._extract_equations(text)),
            "has_tables": bool(self._extract_tables(text))
        }
        
        return metadata
    
    def extract_technical_specifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract technical specifications from text."""
        import re
        
        specifications = []
        
        # Pattern for specifications with context
        spec_patterns = [
            r'([A-Za-z\s]+):\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)',
            r'([A-Za-z\s]+)\s*=\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)',
            r'([A-Za-z\s]+)\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)'
        ]
        
        for pattern in spec_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                specifications.append({
                    "parameter": match.group(1).strip(),
                    "value": float(match.group(2)),
                    "unit": match.group(3),
                    "position": match.start()
                })
        
        return specifications
    
    def extract_materials_list(self, text: str) -> List[Dict[str, Any]]:
        """Extract materials mentioned in the document."""
        import re
        
        materials = []
        material_keywords = [
            'steel', 'aluminum', 'titanium', 'concrete', 'composite', 'polymer',
            'ceramic', 'alloy', 'plastic', 'rubber', 'glass', 'wood', 'carbon',
            'fiber', 'resin', 'epoxy', 'polyethylene', 'polypropylene', 'nylon'
        ]
        
        for material in material_keywords:
            pattern = rf'\b{material}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get context around the material mention
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                materials.append({
                    "material": material,
                    "context": context,
                    "position": match.start()
                })
        
        return materials
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        base_status = super().get_status()
        base_status.update({
            "analyzer_type": "technical",
            "technical_sections": self.technical_sections
        })
        return base_status
