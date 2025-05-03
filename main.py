"""
HR Decision Support System
Combined module for processing documents and generating HR reports
"""

import os
import logging
import tempfile
import json
import argparse
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import random
from pydub import AudioSegment
from pydub.utils import which 

# Document processing libraries
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment
import librosa

# Visualization and reporting
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for API key for LLM features
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. LLM analysis features will be limited.")

# ==================== Document Processing Classes ====================

class PDFExtractor:
    """Extracts text and data from PDF documents"""
    
    def __init__(self, enable_ocr: bool = True):
        self.enable_ocr = enable_ocr
        if enable_ocr:
            try:
                pytesseract.get_tesseract_version()
            except EnvironmentError:
                logger.warning("Tesseract OCR is not installed. OCR functionality will be disabled.")
                self.enable_ocr = False

    def extract_text(self, file_path: str, extract_tables: bool = False) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            return self._extract_with_pdfplumber(file_path, extract_tables)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            if self.enable_ocr:
                logger.info("Attempting OCR extraction")
                return self._extract_with_ocr(file_path)
            return {"success": False, "error": f"Failed to extract text: {str(e)}"}

    def _extract_with_pdfplumber(self, file_path: str, extract_tables: bool) -> Dict[str, Any]:
        text_content = []
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata
            
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
                
                if extract_tables:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            
        return {
            "success": True,
            "text_content": "\n".join(text_content) if text_content else None,
            "tables": tables if tables else None,
            "metadata": metadata
        }

    def _extract_with_ocr(self, file_path: str) -> Dict[str, Any]:
        try:
            import pdf2image
        except ImportError:
            return {
                "success": False,
                "error": "pdf2image package required for OCR. Install with: pip install pdf2image"
            }

        try:
            images = pdf2image.convert_from_path(file_path)
            text_content = []
            
            for image in images:
                text = pytesseract.image_to_string(image)
                text_content.append(text)
            
            return {
                "success": True,
                "text_content": "\n".join(text_content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR extraction failed: {str(e)}"
            }

class ImageExtractor:
    """Extracts text and data from image files"""
    
    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if GROQ_API_KEY:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage, SystemMessage
            return ChatGroq(temperature=0.1, model_name="llama3-8b-8192", api_key=GROQ_API_KEY)
        return None

    def extract_data(self, image_path: str, query: Optional[str] = None) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"error": f"File not found: {image_path}"}
        
        # Try to determine if the image contains handwriting
        contains_handwriting = self._detect_handwriting(image_path)
        
        # Extract text based on content type
        text = self.extract_handwriting(image_path) if contains_handwriting else self.extract_text(image_path)
        
        # Try to identify the document type
        doc_type = self.detect_document_type(image_path)
        
        result = {
            "file_path": image_path,
            "text_content": text,
            "detected_document_type": doc_type,
            "contains_handwriting": contains_handwriting,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # If query is provided, analyze the text
        if query and text and self.llm:
            analysis = self.analyze_with_llm(text, query)
            result["analysis"] = analysis
        
        return result

    def _detect_handwriting(self, image_path: str) -> bool:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            return lines is None or len(lines) < 10
        except Exception:
            return False

    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        try:
            if preprocess:
                processed_img = self._preprocess_image(image_path)
                temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                cv2.imwrite(temp_path, processed_img)
                text = pytesseract.image_to_string(temp_path)
                os.unlink(temp_path)
            else:
                text = pytesseract.image_to_string(image_path)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            return opening
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return cv2.imread(image_path)

    def extract_handwriting(self, image_path: str) -> str:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            custom_config = r'--oem 1 --psm 6'
            text = pytesseract.image_to_string(opening, config=custom_config)
            return text
        except Exception as e:
            logger.error(f"Error extracting handwriting: {str(e)}")
            return ""

    def detect_document_type(self, image_path: str) -> str:
        text = self.extract_text(image_path, preprocess=False).lower()
        
        if any(keyword in text for keyword in ['cv', 'resume', 'curriculum vitae']):
            return 'cv'
        elif any(keyword in text for keyword in ['invoice', 'bill', 'payment']):
            return 'invoice'
        elif any(keyword in text for keyword in ['form', 'application']):
            return 'form'
        elif any(keyword in text for keyword in ['id', 'identification', 'passport', 'license']):
            return 'id_document'
        elif any(keyword in text for keyword in ['certificate', 'diploma']):
            return 'certificate'
        else:
            return 'unknown'

    def analyze_with_llm(self, text: str, query: str) -> Dict[str, Any]:
        if not self.llm:
            return {"error": "LLM not available"}
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_message = SystemMessage(content=f"""
        You are an HR analytics assistant specializing in document analysis. 
        Analyze the provided text extracted from an image and extract information based on the query.
        Provide your response as a structured JSON object with relevant fields. 
        Keep your analysis focused and concise.
        """)
        
        human_message = HumanMessage(content=f"""
        Query: {query}
        
        Text extracted from image:
        {text[:5000]}
        
        Return a JSON object with relevant fields and analysis.
        """)
        
        try:
            response = self.llm.invoke([system_message, human_message])
            content = response.content
            
            if "```json" in content and "```" in content.split("```json")[1]:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content and "```" in content.split("```")[1]:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
                
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error analyzing with LLM: {str(e)}")
            return {"error": str(e)}

class AudioExtractor:
    """Extracts text and features from audio files"""
    
    def __init__(self, language: str = "en-US", use_enhanced: bool = True):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.use_enhanced = use_enhanced
        logger.info(f"AudioExtractor initialized with language={language}")

    def extract_text(self, audio_path: str) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            return {"success": False, "error": f"Audio file not found: {audio_path}"}
        
        try:
            file_extension = os.path.splitext(audio_path)[1].lower()
            
            if file_extension not in ['.wav', '.aiff', '.flac']:
                audio_path = self._convert_to_wav(audio_path)
            
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=self.language)
                features = self._extract_audio_features(audio_path)
                
                return {
                    "success": True,
                    "text": text,
                    "language": self.language,
                    "features": features
                }
                
        except sr.UnknownValueError:
            return {"success": False, "error": "Speech not recognized"}
        except sr.RequestError as e:
            return {"success": False, "error": f"API error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _convert_to_wav(self, audio_path: str) -> str:
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            
            audio = AudioSegment.from_file(audio_path)
            audio.export(temp_path, format="wav")
            return temp_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise
    
    def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        try:
            y, sr = librosa.load(audio_path)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            energy_level = np.mean(librosa.feature.rms(y=y))
            
            return {
                "tempo": float(tempo),
                "zero_crossing_rate": float(zcr),
                "mfcc_mean": [float(x) for x in mfccs],
                "spectral_centroid": float(spectral_centroid),
                "energy_level": float(energy_level)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_interview(self, audio_path: str) -> Dict[str, Any]:
        result = self.extract_text(audio_path)
        
        if not result["success"]:
            return result
        
        text = result["text"]
        features = result["features"]
        
        words = len(text.split())
        speaking_rate = features["tempo"] * words / 60
        
        confidence_score = 0.5
        if features["energy_level"] > 0.1:
            confidence_score += 0.2
        if speaking_rate > 100:
            confidence_score += 0.1
        
        stress_level = min(1.0, features["energy_level"] * 5)
        
        result["analysis"] = {
            "word_count": words,
            "speaking_rate": speaking_rate,
            "estimated_confidence": min(1.0, confidence_score),
            "estimated_stress_level": stress_level
        }
        
        return result
    
    def extract_keywords_for_hr(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        
        hr_keywords = {
            "skills": ["experience", "skill", "ability", "proficient", "expert", "knowledge", 
                      "certified", "qualification", "trained", "competent"],
            "education": ["degree", "university", "college", "diploma", "certificate", 
                         "bachelor", "master", "phd", "graduate", "study"],
            "positions": ["manager", "director", "lead", "supervisor", "executive", 
                         "assistant", "coordinator", "specialist", "officer", "analyst"],
            "soft_skills": ["communication", "teamwork", "leadership", "problem", "solving", 
                           "creative", "interpersonal", "adaptable", "flexible", "motivated"],
            "concerns": ["stress", "overwork", "burnout", "unhappy", "dissatisfied", 
                        "frustrated", "conflict", "issue", "problem", "concern"]
        }
        
        results = {category: [] for category in hr_keywords}
        
        for category, keywords in hr_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    results[category].append(keyword)
        
        return {
            "keywords": results,
            "total_matches": sum(len(matches) for matches in results.values())
        }

# ==================== HR Reporting Classes ====================

class HRReportGenerator:
    """Generates HR reports and statistics"""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.use_demo_data = db_connection is None
        logger.info("HR Report Generator initialized")
    
    def get_pending_decisions(self) -> List[Dict[str, Any]]:
        if self.use_demo_data:
            return self._generate_demo_pending_decisions()
        
        decisions = []
        try:
            # Replace with actual database query
            pass
        except Exception as e:
            logger.error(f"Error retrieving pending decisions: {str(e)}")
            
        return decisions
    
    def generate_hr_report(self, period: str = "last 7 days", department: Optional[str] = None) -> Dict[str, Any]:
        if self.use_demo_data:
            return self._generate_demo_hr_report(period, department)
        
        report_data = {}
        try:
            # Replace with actual database queries
            pass
        except Exception as e:
            logger.error(f"Error generating HR report: {str(e)}")
            
        return report_data
    
    def analyze_employee_attrition_risk(self, employee_id: Optional[str] = None) -> Dict[str, Any]:
        if self.use_demo_data:
            return self._generate_demo_attrition_risk(employee_id)
        
        risk_analysis = {}
        try:
            # Replace with actual database queries and ML model predictions
            pass
        except Exception as e:
            logger.error(f"Error analyzing attrition risk: {str(e)}")
            
        return risk_analysis
    
    def generate_visualization(self, report_data: Dict[str, Any], chart_type: str = "bar", save_path: Optional[str] = None) -> str:
        try:
            plt.figure(figsize=(10, 6))
            
            if chart_type == "bar" and "department_headcount" in report_data:
                depts = list(report_data["department_headcount"].keys())
                counts = list(report_data["department_headcount"].values())
                plt.bar(depts, counts)
                plt.title("Headcount by Department")
                plt.xlabel("Department")
                plt.ylabel("Employee Count")
            
            elif chart_type == "line" and "attrition_trend" in report_data:
                dates = list(report_data["attrition_trend"].keys())
                values = list(report_data["attrition_trend"].values())
                plt.plot(dates, values)
                plt.title("Attrition Trend")
                plt.xlabel("Date")
                plt.ylabel("Attrition Rate (%)")
                plt.xticks(rotation=45)
                
            elif chart_type == "pie" and "absence_reasons" in report_data:
                reasons = list(report_data["absence_reasons"].keys())
                counts = list(report_data["absence_reasons"].values())
                plt.pie(counts, labels=reasons, autopct='%1.1f%%')
                plt.title("Absence Reasons Distribution")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                return save_path
            else:
                temp_path = f"temp_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(temp_path)
                return temp_path
                
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return ""
        
    def export_report_to_json(self, report_data: Dict[str, Any], 
                            file_path: str) -> bool:
        try:
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error exporting report to JSON: {str(e)}")
            return False
    
    def export_report_to_csv(self, report_data: Dict[str, Any], 
                           file_path: str) -> bool:
        try:
            flattened_data = self._flatten_report_data(report_data)
            df = pd.DataFrame(flattened_data)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting report to CSV: {str(e)}")
            return False
    
    def _flatten_report_data(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        flattened = []
        
        if "department_headcount" in report_data:
            for dept, count in report_data["department_headcount"].items():
                flattened.append({
                    "Metric": "Department Headcount",
                    "Category": dept,
                    "Value": count
                })
                
        if "attrition_risk" in report_data:
            for risk_level, count in report_data["attrition_risk"].items():
                flattened.append({
                    "Metric": "Attrition Risk",
                    "Category": risk_level,
                    "Value": count
                })
        
        return flattened
    
    # --- Demo data generation methods ---
    
    def _generate_demo_pending_decisions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "dec_001",
                "action_name": "evaluate_leave_request",
                "parameters": {
                    "employee_email": "john.doe@example.com",
                    "details": "Annual leave from June 5-12",
                    "task_id": "task_001"
                },
                "confidence_score": 0.85,
                "justification": "Employee has accumulated sufficient leave days. Team coverage available.",
                "status": "Pending Validation",
                "timestamp": datetime.now().timestamp()
            },
            {
                "id": "dec_002",
                "action_name": "evaluate_promotion_request",
                "parameters": {
                    "employee_email": "jane.smith@example.com",
                    "details": "Promotion to Senior Developer",
                    "task_id": "task_002"
                },
                "confidence_score": 0.65,
                "justification": "Employee has met performance metrics. Position available in budget.",
                "status": "Pending Validation",
                "timestamp": datetime.now().timestamp()
            }
        ]
    
    def _generate_demo_hr_report(self, period: str, department: Optional[str] = None) -> Dict[str, Any]:
        today = datetime.now()
        if period == "last 7 days":
            start_date = today - timedelta(days=7)
        elif period == "last 30 days":
            start_date = today - timedelta(days=30)
        elif period == "current month":
            start_date = today.replace(day=1)
        else:
            start_date = today - timedelta(days=7)
            
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = today.strftime("%Y-%m-%d")
        
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
        if department:
            departments = [d for d in departments if d.lower() == department.lower()]
            
        department_headcount = {dept: random.randint(5, 50) for dept in departments}
        
        attrition_risk = {
            "High Risk": random.randint(1, 5),
            "Medium Risk": random.randint(5, 15),
            "Low Risk": random.randint(40, 100)
        }
        
        absence_reasons = {
            "Sick Leave": random.randint(5, 20),
            "Vacation": random.randint(10, 30),
            "Family Emergency": random.randint(1, 5),
            "Other": random.randint(1, 10)
        }
        
        attrition_trend = {}
        for i in range(6):
            month_date = today - timedelta(days=30 * i)
            month_str = month_date.strftime("%b %Y")
            attrition_trend[month_str] = round(random.uniform(1.0, 5.0), 1)
        
        return {
            "report_period": f"{start_date_str} to {end_date_str}",
            "generation_date": today.strftime("%Y-%m-%d %H:%M:%S"),
            "department_filter": department,
            "summary": {
                "total_employees": sum(department_headcount.values()),
                "new_hires": random.randint(1, 10),
                "terminations": random.randint(0, 5),
                "open_positions": random.randint(2, 8),
                "average_tenure_years": round(random.uniform(1.5, 5.0), 1)
            },
            "department_headcount": department_headcount,
            "attrition_risk": attrition_risk,
            "absence_reasons": absence_reasons,
            "attrition_trend": attrition_trend,
            "pending_requests": random.randint(3, 15),
            "training_completion_rate": f"{random.randint(70, 95)}%"
        }
    
    def _generate_demo_attrition_risk(self, employee_id: Optional[str] = None) -> Dict[str, Any]:
        if employee_id:
            risk_score = round(random.uniform(0.1, 0.9), 2)
            risk_level = "Low"
            if risk_score > 0.7:
                risk_level = "High"
            elif risk_score > 0.4:
                risk_level = "Medium"
                
            possible_factors = [
                ("Compensation", random.uniform(-0.3, 0.3)),
                ("Work-Life Balance", random.uniform(-0.3, 0.3)),
                ("Career Growth", random.uniform(-0.3, 0.3)),
                ("Manager Relationship", random.uniform(-0.3, 0.3)),
                ("Job Satisfaction", random.uniform(-0.3, 0.3))
            ]
            
            sorted_factors = sorted(possible_factors, key=lambda x: abs(x[1]), reverse=True)[:3]
            factors = []
            
            for factor, impact in sorted_factors:
                impact_direction = "positive" if impact > 0 else "negative"
                factors.append({
                    "factor": factor,
                    "impact": abs(impact),
                    "direction": impact_direction
                })
            
            return {
                "employee_id": employee_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "top_factors": factors,
                "recommendation": self._get_recommendation(risk_level)
            }
        else:
            return {
                "summary": {
                    "high_risk_count": random.randint(1, 5),
                    "medium_risk_count": random.randint(5, 15),
                    "low_risk_count": random.randint(40, 100),
                    "average_risk_score": round(random.uniform(0.2, 0.4), 2),
                    "trend": random.choice(["Stable", "Improving", "Declining"])
                },
                "departments_at_risk": [
                    {
                        "department": "Sales",
                        "risk_score": round(random.uniform(0.4, 0.7), 2),
                        "affected_headcount": random.randint(2, 8)
                    },
                    {
                        "department": "Engineering",
                        "risk_score": round(random.uniform(0.3, 0.6), 2),
                        "affected_headcount": random.randint(1, 5)
                    }
                ],
                "recommendations": [
                    "Schedule skip-level meetings with high-risk teams",
                    "Review compensation packages for key roles",
                    "Implement mentorship program for career development"
                ]
            }
    
    def _get_recommendation(self, risk_level: str) -> str:
        recommendations = {
            "High": [
                "Schedule immediate 1:1 meeting to discuss concerns",
                "Review compensation package and career path",
                "Consider retention bonus or special project assignment"
            ],
            "Medium": [
                "Schedule regular check-ins with manager",
                "Identify growth opportunities within current role",
                "Provide additional training or development options"
            ],
            "Low": [
                "Maintain current engagement strategies",
                "Continue regular performance reviews",
                "Consider for mentoring opportunities"
            ]
        }
        
        return random.choice(recommendations.get(risk_level, ["No specific recommendation"]))

# ==================== Main Application Functions ====================

def process_document(file_path: str, doc_type: str = None, query: str = None) -> Dict[str, Any]:
    """Process different document types and extract relevant information"""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    if doc_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.pdf']:
            doc_type = 'pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            doc_type = 'image'
        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
            doc_type = 'audio'
        else:
            doc_type = 'text'
    
    if doc_type == 'pdf':
        extractor = PDFExtractor()
        result = extractor.extract_text(file_path)
        if query and result.get("text_content"):
            # For PDFs, we can add additional analysis here if needed
            pass
        return result
    elif doc_type == 'image':
        extractor = ImageExtractor()
        return extractor.extract_data(file_path, query)
    elif doc_type == 'audio':
        extractor = AudioExtractor()
        if query and "interview" in query.lower():
            return extractor.analyze_interview(file_path)
        else:
            result = extractor.extract_text(file_path)
            if result["success"] and "text" in result:
                keywords = extractor.extract_keywords_for_hr(result["text"])
                result["keywords"] = keywords
            return result
    else:
        return {"error": f"Unsupported document type: {doc_type}"}

def generate_hr_report(report_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate different types of HR reports"""
    report_gen = HRReportGenerator()
    
    if report_type == 'pending_decisions':
        return {"pending_decisions": report_gen.get_pending_decisions()}
    elif report_type == 'hr_summary':
        period = params.get("period", "last 7 days") if params else "last 7 days"
        department = params.get("department") if params else None
        return report_gen.generate_hr_report(period, department)
    elif report_type == 'attrition_risk':
        employee_id = params.get("employee_id") if params else None
        return report_gen.analyze_employee_attrition_risk(employee_id)
    else:
        return {"error": f"Unsupported report type: {report_type}"}

def analyze_cv(cv_path: str) -> Dict[str, Any]:
    """Analyze a CV document and extract relevant information"""
    cv_data = process_document(cv_path, "pdf", "Extract key information including name, education, experience, skills, and contact details")
    
    return {
        "cv_data": cv_data,
        "analysis": {
            "skills_match": cv_data.get("skills_analysis", {}),
            "experience_years": cv_data.get("experience_years", 0),
            "education_level": cv_data.get("education_level", ""),
            "recommendation": cv_data.get("recommendation", "")
        }
    }

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='HR Analytics System')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Document processing parser
    doc_parser = subparsers.add_parser('process', help='Process a document')
    doc_parser.add_argument('file_path', help='Path to the document')
    doc_parser.add_argument('--type', choices=['pdf', 'image', 'audio'], help='Document type')
    doc_parser.add_argument('--query', help='Query to guide extraction')
    
    # Report generation parser
    report_parser = subparsers.add_parser('report', help='Generate an HR report')
    report_parser.add_argument('report_type', choices=['pending_decisions', 'hr_summary', 'attrition_risk'], help='Type of report to generate')
    report_parser.add_argument('--employee_id', help='Employee ID for attrition risk report')
    report_parser.add_argument('--period', help='Period for HR summary report (e.g., "last 7 days")')
    report_parser.add_argument('--department', help='Department filter for HR summary report')
    
    # CV analysis parser
    cv_parser = subparsers.add_parser('analyze_cv', help='Analyze a CV document')
    cv_parser.add_argument('cv_path', help='Path to the CV document')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        result = process_document(args.file_path, args.type, args.query)
        print(json.dumps(result, indent=2))
    elif args.command == 'report':
        params = {}
        if args.employee_id:
            params["employee_id"] = args.employee_id
        if args.period:
            params["period"] = args.period
        if args.department:
            params["department"] = args.department
        result = generate_hr_report(args.report_type, params)
        print(json.dumps(result, indent=2))
    elif args.command == 'analyze_cv':
        result = analyze_cv(args.cv_path)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()