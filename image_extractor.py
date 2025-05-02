"""
Image Extractor Module - Extract text from images using OCR with handwriting recognition
"""
import logging
import re
import time
import os
import io
from typing import Dict, Any, List, Optional, Union
import base64

# Import necessary libraries
try:
    import pytesseract # type: ignore
    from PIL import Image
    import cv2
    import numpy as np
    from difflib import get_close_matches
    
    # Check if we have all required packages
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    missing_package = str(e).split("'")[1]
    logging.warning(f"OCR functionality limited: Missing package {missing_package}. "
                    f"Install with 'pip install {missing_package}'")

# Configure logging
logger = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self, db_instance):
        """
        Initialize with a database connection
        
        Args:
            db_instance: Database instance to store logs and results
        """
        self.db = db_instance
        self.extracted_images = {}  # Cache for extracted text
        self.ocr_available = OCR_AVAILABLE
        
        # Initialize OCR if available
        if self.ocr_available:
            # Check if Tesseract is properly installed
            try:
                pytesseract.get_tesseract_version()
                logger.info("OCR (Tesseract) is available")
            except Exception as e:
                self.ocr_available = False
                logger.warning(f"Tesseract not properly configured: {str(e)}")
                logger.warning("Make sure Tesseract is installed and in PATH")
        else:
            logger.warning("OCR functionality not available. Install required packages.")
        
        logger.info("Image Extractor initialized")
    
    def extract_text_from_image(self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None, is_handwritten: bool = False) -> Dict[str, Any]:
        """
        Extract text from an image using OCR
        
        Args:
            image_path: Path to image file (optional)
            image_bytes: Image as bytes (optional)
            is_handwritten: Whether to optimize for handwritten text
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        if not self.ocr_available:
            return {
                "error": "OCR functionality not available. Install required packages.",
                "text": "",
                "confidence": 0
            }
        
        logger.info(f"Extracting text from image (handwritten={is_handwritten})")
        
        try:
            # Load image either from path or bytes
            if image_path and os.path.exists(image_path):
                image = cv2.imread(image_path)
                source_name = os.path.basename(image_path)
            elif image_bytes:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                source_name = f"image_{int(time.time())}"
            else:
                return {
                    "error": "No valid image source provided",
                    "text": "",
                    "confidence": 0
                }
            
            # Preprocess the image
            processed_image = self._preprocess_image(image, is_handwritten)
            
            # Perform OCR
            ocr_config = '--oem 3 --psm 6'
            if is_handwritten:
                # Use a different PSM mode for handwritten text
                ocr_config = '--oem 3 --psm 12'
            
            # Get OCR data with confidence
            ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=ocr_config)
            
            # Extract text and confidence values for each word
            text_parts = []
            confidences = []
            
            for i in range(len(ocr_data['text'])):
                # Skip empty words
                if ocr_data['text'][i].strip():
                    text_parts.append(ocr_data['text'][i])
                    confidences.append(float(ocr_data['conf'][i]))
            
            # Combine text parts
            extracted_text = ' '.join(text_parts)
            
            # Calculate average confidence for non-zero confidence values
            valid_confidences = [c for c in confidences if c > 0]
            avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0
            
            # Clean up the text
            extracted_text = self._clean_extracted_text(extracted_text, is_handwritten)
            
            # Store in cache
            self.extracted_images[source_name] = {
                "text": extracted_text,
                "confidence": avg_confidence,
                "timestamp": time.time()
            }
            
            # Log the extraction
            self.db["agent_log"].append({
                "action": "extract_text_from_image", 
                "params": {
                    "source": "file" if image_path else "bytes", 
                    "is_handwritten": is_handwritten
                }, 
                "timestamp": time.time()
            })
            
            return {
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": len(text_parts),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return {
                "error": f"Error extracting text: {str(e)}",
                "text": "",
                "confidence": 0
            }
    
    def process_form_image(self,image_path: Optional[str] = None, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Process an image of a form and extract structured data
        
        Args:
            image_path: Path to form image file (optional)
            image_bytes: Form image as bytes (optional)
            
        Returns:
            Dictionary with structured form data
        """
        if not self.ocr_available:
            return {"error": "OCR functionality not available"}
        
        logger.info("Processing form image")
        
        # Extract text first
        extraction_result = self.extract_text_from_image(
            image_path=image_path,
            image_bytes=image_bytes,
            is_handwritten=False  # Forms are usually printed
        )
        
        if "error" in extraction_result:
            return extraction_result
        
        extracted_text = extraction_result["text"]
        
        # Process the form by looking for common patterns
        form_data = self._extract_form_fields(extracted_text)
        
        # Log the form processing
        self.db["agent_log"].append({
            "action": "process_form_image", 
            "params": {"fields_extracted": len(form_data)}, 
            "timestamp": time.time()
        })
        
        return {
            "form_data": form_data,
            "raw_text": extracted_text,
            "confidence": extraction_result["confidence"],
            "timestamp": time.time()
        }
    
    def extract_id_document(self,image_path: Optional[str] = None,image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Extract information from ID cards or documents
        
        Args:
            image_path: Path to ID document image file (optional)
            image_bytes: ID document image as bytes (optional)
            
        Returns:
            Dictionary with structured ID information
        """
        if not self.ocr_available:
            return {"error": "OCR functionality not available"}
        
        logger.info("Extracting ID document information")
        
        # Extract text first
        extraction_result = self.extract_text_from_image(
            image_path=image_path,
            image_bytes=image_bytes,
            is_handwritten=False
        )
        
        if "error" in extraction_result:
            return extraction_result
        
        extracted_text = extraction_result["text"]
        
        # Process the ID by looking for specific patterns
        id_data = self._extract_id_fields(extracted_text)
        
        # Log the ID processing
        self.db["agent_log"].append({
            "action": "extract_id_document", 
            "params": {"fields_extracted": len(id_data)}, 
            "timestamp": time.time()
        })
        
        return {
            "id_data": id_data,
            "raw_text": extracted_text,
            "confidence": extraction_result["confidence"],
            "timestamp": time.time()
        }
    
    def analyze_handwritten_note(self,image_path: Optional[str] = None,image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Analyze handwritten notes and extract meaning
        
        Args:
            image_path: Path to handwritten note image (optional)
            image_bytes: Handwritten note image as bytes (optional)
            
        Returns:
            Dictionary with note content and sentiment analysis
        """
        if not self.ocr_available:
            return {"error": "OCR functionality not available"}
        
        logger.info("Analyzing handwritten note")
        
        # Extract text with handwriting optimization
        extraction_result = self.extract_text_from_image(
            image_path=image_path,
            image_bytes=image_bytes,
            is_handwritten=True
        )
        
        if "error" in extraction_result:
            return extraction_result
        
        extracted_text = extraction_result["text"]
        
        # Perform basic sentiment analysis
        sentiment = self._analyze_text_sentiment(extracted_text)
        
        # Extract key information
        key_info = self._extract_key_info(extracted_text)
        
        # Log the note analysis
        self.db["agent_log"].append({
            "action": "analyze_handwritten_note", 
            "params": {"sentiment": sentiment["sentiment"]}, 
            "timestamp": time.time()
        })
        
        return {
            "text": extracted_text,
            "confidence": extraction_result["confidence"],
            "sentiment": sentiment,
            "key_information": key_info,
            "timestamp": time.time()
        }
    
    def create_document_decision(self, document_data: Dict[str, Any], analysis_type: str) -> str:
        """
        Create an AI decision record based on document analysis
        
        Args:
            document_data: Extracted document data
            analysis_type: Type of document analysis
            
        Returns:
            Decision ID
        """
        logger.info(f"Creating document decision for {analysis_type}")
        
        # Base confidence on OCR confidence
        confidence_score = min(document_data.get("confidence", 0) / 100, 0.95)
        
        # Generate justification based on analysis type
        justification = "Document processed successfully"
        parameters = {"analysis_type": analysis_type}
        
        if analysis_type == "id_verification":
            parameters["document_type"] = "ID Document"
            parameters["fields_found"] = list(document_data.get("id_data", {}).keys())
            parameters["verification_status"] = "Pending HR Review"
            justification = "ID document processed with OCR. Manual verification required."
            
        elif analysis_type == "handwritten_note":
            parameters["sentiment"] = document_data.get("sentiment", {}).get("sentiment", "neutral")
            parameters["key_points"] = document_data.get("key_information", [])
            justification = f"Handwritten note processed. Sentiment: {parameters['sentiment']}"
            
        elif analysis_type == "form_processing":
            parameters["form_fields"] = document_data.get("form_data", {})
            parameters["processing_status"] = "Completed" if parameters["form_fields"] else "Incomplete"
            justification = f"Form processed with {len(parameters['form_fields'])} fields extracted"
        
        # Use the existing API to create a decision
        from main import create_ai_decision_impl
        
        decision_result = create_ai_decision_impl(
            action_name=f"process_{analysis_type}_document",
            parameters=parameters,
            confidence_score=confidence_score,
            justification=justification
        )
        
        # Log the decision creation
        self.db["agent_log"].append({
            "action": "create_document_decision", 
            "params": {"analysis_type": analysis_type}, 
            "timestamp": time.time()
        })
        
        return decision_result
    
    def _preprocess_image(self, image: np.ndarray, is_handwritten: bool = False) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: OpenCV image
            is_handwritten: Whether to optimize for handwritten text
            
        Returns:
            Processed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if is_handwritten:
            # Special processing for handwritten text
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Invert back
            result = cv2.bitwise_not(opening)
        else:
            # Processing for printed text
            # Noise removal with bilateral filter
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply threshold to get black and white image
            _, result = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return result
    
    def _clean_extracted_text(self, text: str, is_handwritten: bool = False) -> str:
        """
        Clean up extracted OCR text
        
        Args:
            text: Raw extracted text
            is_handwritten: Whether text is from handwritten source
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        if is_handwritten:
            # For handwritten text, try to correct common OCR errors
            # This is a simplified version - in a production system,
            # you would use a more sophisticated spell checker
            
            # Replace common OCR errors
            replacements = {
                '0': 'O',
                '1': 'I',
                '5': 'S',
                '8': 'B',
                '@': 'a',
            }
            
            for old, new in replacements.items():
                # Only replace if it seems to be an error (not part of a number)
                text = text.replace(f' {old} ', f' {new} ')
            
            # Clean up any remaining issues
            text = text.replace('  ', ' ')
        
        return text
    
    def _extract_form_fields(self, text: str) -> Dict[str, str]:
        """
        Extract form fields from extracted text
        
        Args:
            text: Extracted text from form
            
        Returns:
            Dictionary of field names and values
        """
        form_data = {}
        
        # Look for patterns like "Field: Value" or "Field - Value"
        field_patterns = [
            r'([A-Za-z\s]+):\s*([A-Za-z0-9\s]+)',  # Field: Value
            r'([A-Za-z\s]+)-\s*([A-Za-z0-9\s]+)',  # Field - Value
            r'([A-Za-z\s]+)\s{2,}([A-Za-z0-9\s]+)'  # Field    Value
        ]
        
        # Common form fields to look for
        common_fields = [
            "name", "date", "address", "phone", "email", 
            "signature", "id", "social security", "ssn",
            "date of birth", "gender", "position", "department"
        ]
        
        # Apply each pattern
        for pattern in field_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                field = match.group(1).strip().lower()
                value = match.group(2).strip()
                
                # Store if not empty
                if value:
                    form_data[field] = value
        
        # Look for common fields that might not follow the patterns
        lines = text.split('\n')
        for field in common_fields:
            if field not in form_data:
                for i, line in enumerate(lines):
                    if field.lower() in line.lower() and i+1 < len(lines):
                        # Check if the next line might be the value
                        next_line = lines[i+1].strip()
                        if next_line and not any(f.lower() in next_line.lower() for f in common_fields):
                            form_data[field] = next_line
        
        return form_data
    
    def _extract_id_fields(self, text: str) -> Dict[str, str]:
        """
        Extract common ID fields from extracted text
        
        Args:
            text: Extracted text from ID document
            
        Returns:
            Dictionary of ID fields
        """
        id_data = {}
        
        # Define patterns for common ID fields
        patterns = {
            "name": r'(?:name|full name|nombre)[\s:]+([A-Za-z\s]+)',
            "id_number": r'(?:id|identification|number|#)[\s:]+([A-Z0-9-]+)',
            "dob": r'(?:birth|dob|date of birth|born)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{2,4})',
            "expiration": r'(?:expiration|exp|expires|valid until)[\s:]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            "address": r'(?:address|addr)[\s:]+([A-Za-z0-9\s,]+)',
            "gender": r'(?:gender|sex)[\s:]+([MF]|Male|Female)',
        }
        
        # Apply each pattern
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                id_data[field] = match.group(1).strip()
        
        return id_data
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform basic sentiment analysis on text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        # This is a very basic sentiment analysis
        # In a production system, you would use a proper NLP library
        
        # Lists of positive and negative words
        positive_words = [
            'good', 'great', 'excellent', 'happy', 'positive', 'pleased',
            'satisfied', 'wonderful', 'fantastic', 'approve', 'approved',
            'accept', 'accepted', 'thank', 'thanks', 'appreciate', 'yes'
        ]
        
        negative_words = [
            'bad', 'poor', 'terrible', 'unhappy', 'negative', 'displeased',
            'dissatisfied', 'awful', 'horrible', 'reject', 'rejected',
            'decline', 'declined', 'sorry', 'regret', 'unfortunately', 'no'
        ]
        
        # Count occurrences
        text_lower = text.lower()
        positive_count = sum(text_lower.count(' ' + word + ' ') for word in positive_words)
        negative_count = sum(text_lower.count(' ' + word + ' ') for word in negative_words)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(0.5 + (positive_count - negative_count) * 0.1, 1.0)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(0.5 - (negative_count - positive_count) * 0.1, 0.0)
        else:
            sentiment = "neutral"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_count": positive_count,
            "negative_count": negative_count
        }
    
    def _extract_key_info(self, text: str) -> List[str]:
        """
        Extract key information from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key information points
        """
        key_info = []
        
        # Look for dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            key_info.append(f"Date mentioned: {dates[0]}")
        
        # Look for potential names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        names = re.findall(name_pattern, text)
        if names:
            key_info.append(f"Name mentioned: {names[0]}")
        
        # Look for numbers (could be phone, ID, etc.)
        number_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        numbers = re.findall(number_pattern, text)
        if numbers:
            key_info.append(f"Number found: {numbers[0]}")
        
        # Look for keywords
        keywords = ["meeting", "request", "approve", "reject", "schedule", 
                "urgent", "important", "review", "signature", "sign", 
                "leave", "vacation", "sick", "absence"]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                key_info.append(f"Contains keyword: {keyword}")
                break
        
        return key_info