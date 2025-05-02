"""
PDF Extraction Module - Extracts and processes text from PDF documents
Uses PyPDF2 and NLTK for text extraction and processing, with CV parsing capabilities
"""
import logging
import time
import re
import os
from typing import Dict, Any, List, Optional
import io

# Import necessary libraries
import PyPDF2 # type: ignore
import nltk # type: ignore
from nltk.tokenize import word_tokenize, sent_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore

# For first-time setup, uncomment these lines:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Optional: For more advanced NLP
try:
    import spacy # type: ignore
    NLP_LOADED = True
    # Load English language model (you'll need to install it first)
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    NLP_LOADED = False

# Configure logging
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, db_instance):
        """Initialize with a database connection"""
        self.db = db_instance
        self.extracted_documents = {}  # Cache for extracted text
        logger.info("PDF Extractor initialized")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract raw text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return "Error: File not found"
        
        try:
            # Open and read PDF file
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from each page
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                # Cache the extracted text
                doc_id = os.path.basename(file_path)
                self.extracted_documents[doc_id] = text
                
                # Log the extraction
                self.db["agent_log"].append({
                    "action": "extract_text_from_pdf", 
                    "params": {"file_path": file_path}, 
                    "timestamp": time.time()
                })
                
                return text
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text: {str(e)}"
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes (for uploaded files)
        
        Args:
            pdf_bytes: PDF file content as bytes
            
        Returns:
            Extracted text content as a string
        """
        logger.info("Extracting text from PDF bytes")
        
        try:
            # Read PDF from bytes
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            # Generate a unique ID for this extraction
            import uuid
            doc_id = f"upload_{str(uuid.uuid4())[:8]}"
            self.extracted_documents[doc_id] = text
            
            # Log the extraction
            self.db["agent_log"].append({
                "action": "extract_text_from_pdf_bytes", 
                "params": {"bytes_length": len(pdf_bytes)}, 
                "timestamp": time.time()
            })
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {str(e)}")
            return f"Error extracting text: {str(e)}"
    
    def parse_cv(self, text: str = None, file_path: str = None, pdf_bytes: bytes = None) -> Dict[str, Any]:
        """
        Extract structured information from a CV/resume
        
        Args:
            text: Pre-extracted text (optional)
            file_path: Path to PDF file (optional)
            pdf_bytes: PDF bytes (optional)
            
        Returns:
            Dictionary with structured CV information
        """
        logger.info("Parsing CV")
        
        # Get text if not provided
        if text is None:
            if file_path:
                text = self.extract_text_from_pdf(file_path)
            elif pdf_bytes:
                text = self.extract_text_from_pdf_bytes(pdf_bytes)
            else:
                return {"error": "No input provided"}
        
        # Basic information extraction
        cv_data = {
            "personal_info": self._extract_personal_info(text),
            "education": self._extract_education(text),
            "experience": self._extract_experience(text),
            "skills": self._extract_skills(text),
            "languages": self._extract_languages(text),
            "raw_text": text
        }
        
        # Log the parsing action
        self.db["agent_log"].append({
            "action": "parse_cv", 
            "params": {
                "file_path": file_path if file_path else "Bytes or text provided"
            }, 
            "timestamp": time.time()
        })
        
        return cv_data
    
    def analyze_cv_for_role(self, cv_data: Dict[str, Any], role_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CV suitability for a specific role
        
        Args:
            cv_data: Parsed CV data 
            role_requirements: Job requirements
            
        Returns:
            Analysis with match scores and recommendations
        """
        logger.info(f"Analyzing CV for role")
        
        # Extract required skills from role
        required_skills = role_requirements.get("required_skills", [])
        preferred_skills = role_requirements.get("preferred_skills", [])
        
        # Extract candidate skills
        candidate_skills = cv_data.get("skills", [])
        
        # Calculate match scores
        required_matches = [skill for skill in required_skills if self._skill_match(skill, candidate_skills)]
        preferred_matches = [skill for skill in preferred_skills if self._skill_match(skill, candidate_skills)]
        
        required_score = len(required_matches) / len(required_skills) if required_skills else 0
        preferred_score = len(preferred_matches) / len(preferred_skills) if preferred_skills else 0
        
        # Experience match
        experience_match = self._analyze_experience_match(
            cv_data.get("experience", []), 
            role_requirements.get("min_years_experience", 0)
        )
        
        # Education match
        education_match = self._analyze_education_match(
            cv_data.get("education", []),
            role_requirements.get("education_requirements", {})
        )
        
        # Calculate overall score (weighted)
        overall_score = (
            required_score * 0.5 + 
            preferred_score * 0.2 + 
            experience_match * 0.2 + 
            education_match * 0.1
        )
        
        # Generate recommendations
        recommendations = []
        if required_score < 0.8:
            recommendations.append("Candidate lacks some required skills")
        
        if experience_match < 0.7:
            recommendations.append("Experience level may be insufficient")
            
        if education_match < 0.7:
            recommendations.append("Education may not match requirements")
            
        if overall_score > 0.8:
            recommendations.append("Strong overall match, recommend interview")
        elif overall_score > 0.6:
            recommendations.append("Moderate match, may be worth considering")
        else:
            recommendations.append("Low match, not recommended for this role")
        
        # Create decision data
        decision_data = {
            "overall_match_score": round(overall_score, 2),
            "required_skills_score": round(required_score, 2),
            "preferred_skills_score": round(preferred_score, 2),
            "experience_match_score": round(experience_match, 2),
            "education_match_score": round(education_match, 2),
            "matched_required_skills": required_matches,
            "matched_preferred_skills": preferred_matches,
            "missing_required_skills": [s for s in required_skills if s not in required_matches],
            "recommendations": recommendations,
            "timestamp": time.time()
        }
        
        # Log the analysis
        self.db["agent_log"].append({
            "action": "analyze_cv_for_role", 
            "params": {"overall_score": overall_score}, 
            "timestamp": time.time()
        })
        
        return decision_data
    
    def create_automated_cv_decision(self, cv_data: Dict[str, Any], 
                                role_data: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> str:
        """
        Create an automated decision record for CV evaluation
        
        Args:
            cv_data: Parsed CV data
            role_data: Role requirements
            analysis: CV analysis results
            
        Returns:
            Decision ID
        """
        logger.info("Creating automated CV evaluation decision")
        
        # Determine confidence based on match score
        confidence_score = analysis.get("overall_match_score", 0)
        
        # Generate justification
        recommendations = analysis.get("recommendations", [])
        justification = "; ".join(recommendations)
        
        # Determine applicant name
        applicant_name = cv_data.get("personal_info", {}).get("name", "Unknown Applicant")
        
        # Create decision parameters
        parameters = {
            "applicant_name": applicant_name,
            "role_title": role_data.get("title", "Unspecified Role"),
            "match_scores": {
                "overall": analysis.get("overall_match_score", 0),
                "required_skills": analysis.get("required_skills_score", 0),
                "experience": analysis.get("experience_match_score", 0)
            },
            "missing_skills": analysis.get("missing_required_skills", []),
            "recommendation": recommendations[0] if recommendations else "No specific recommendation"
        }
        
        # Use the existing API to create a decision
        from main import create_ai_decision_impl
        
        decision_result = create_ai_decision_impl(
            action_name="evaluate_job_application",
            parameters=parameters,
            confidence_score=confidence_score,
            justification=justification
        )
        
        # Log the decision creation
        self.db["agent_log"].append({
            "action": "create_automated_cv_decision", 
            "params": {"applicant": applicant_name}, 
            "timestamp": time.time()
        })
        
        return decision_result
    
    def _extract_personal_info(self, text: str) -> Dict[str, Any]:
        """Extract personal information from CV text"""
        # Basic extraction with regex patterns
        name_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)'
        email_pattern = r'[\w\.-]+@[\w\.-]+'
        phone_pattern = r'(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}'
        
        # Find first name (assume it's at the beginning)
        name_match = re.search(name_pattern, text[:200])
        name = name_match.group(0) if name_match else "Unknown"
        
        # Find email
        email_match = re.search(email_pattern, text)
        email = email_match.group(0) if email_match else ""
        
        # Find phone
        phone_match = re.search(phone_pattern, text)
        phone = phone_match.group(0) if phone_match else ""
        
        return {
            "name": name,
            "email": email,
            "phone": phone
        }
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from CV text"""
        education = []
        
        # Look for education section
        education_text = self._extract_section(text, ["education", "academic background"])
        if not education_text:
            return education
            
        # Look for degree indicators
        degree_patterns = [
            r'(Bachelor|Master|PhD|Doctorate|B\.S\.|M\.S\.|B\.A\.|M\.A\.|Ph\.D\.)',
            r'(Bachelor of|Master of|Doctor of)'
        ]
        
        lines = education_text.split('\n')
        current_entry = {}
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check for degree
            for pattern in degree_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Save previous entry if exists
                    if current_entry and "degree" in current_entry:
                        education.append(current_entry)
                    
                    # Start new entry
                    current_entry = {
                        "degree": line.strip(),
                        "institution": "",
                        "year": ""
                    }
                    
                    # Try to extract year
                    year_match = re.search(r'(19|20)\d{2}', line)
                    if year_match:
                        current_entry["year"] = year_match.group(0)
                    break
            
            # If no degree found but we have a current entry, it might be institution
                elif current_entry and "degree" in current_entry and not current_entry["institution"]:
                    current_entry["institution"] = line.strip()
        
        # Add the last entry if exists
        if current_entry and "degree" in current_entry:
            education.append(current_entry)
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from CV text"""
        experience = []
        
        # Look for experience section
        experience_text = self._extract_section(text, ["experience", "employment", "work history"])
        if not experience_text:
            return experience
            
        # Try to split into entries
        # This is a simplistic approach; in practice, you'd need more robust parsing
        entries = re.split(r'\n\s*\n', experience_text)
        
        for entry in entries:
            if not entry.strip():
                continue
                
            # Try to extract job title
            lines = entry.split('\n')
            if not lines:
                continue
                
            job_entry = {
                "title": lines[0].strip(),
                "company": lines[1].strip() if len(lines) > 1 else "",
                "duration": "",
                "description": ""
            }
            
            # Try to extract dates
            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\.?\s+\d{4}\s+(\-|to)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December|Present)\.?\s+\d{0,4}'
            date_match = re.search(date_pattern, entry, re.IGNORECASE)
            if date_match:
                job_entry["duration"] = date_match.group(0)
            
            # Extract description (everything else)
            description_lines = lines[2:] if len(lines) > 2 else []
            job_entry["description"] = '\n'.join(description_lines)
            
            experience.append(job_entry)
        
        return experience
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from CV text"""
        skills = []
        
        # Look for skills section
        skills_text = self._extract_section(text, ["skills", "technical skills", "competencies"])
        if not skills_text:
            # Try to extract skills from the whole text
            skills_text = text
        
        # Use more advanced NLP if available
        if NLP_LOADED:
            doc = nlp(skills_text)
            
            # Extract noun phrases as potential skills
            for chunk in doc.noun_chunks:
                skills.append(chunk.text.strip())
            
            # Remove duplicates and filter by length
            skills = list(set([s for s in skills if len(s.split()) <= 3]))
        else:
            # Fallback to simpler extraction
            # Common programming languages and skills
            common_skills = [
                "Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP",
                "HTML", "CSS", "SQL", "NoSQL", "React", "Angular", "Vue", "Node.js",
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git", "Agile", "Scrum",
                "Machine Learning", "Data Science", "AI", "DevOps", "Cloud", "Security"
            ]
            
            # Check for skills in text
            for skill in common_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', skills_text, re.IGNORECASE):
                    skills.append(skill)
            
            # Add potential skills from bullet points
            bullet_matches = re.findall(r'[•\-\*]\s*([^\n\-•\*]+)', skills_text)
            for match in bullet_matches:
                if len(match.split()) <= 3:  # Likely a skill if short
                    skills.append(match.strip())
        
        return skills
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract language skills from CV text"""
        languages = []
        
        # Common languages
        common_languages = [
            "English", "Spanish", "French", "German", "Chinese", "Japanese",
            "Russian", "Arabic", "Portuguese", "Italian", "Dutch", "Korean",
            "Hindi", "Bengali", "Urdu", "Swedish", "Norwegian", "Danish", "Finnish"
        ]
        
        # Look for languages section
        languages_text = self._extract_section(text, ["languages", "language skills"])
        if not languages_text:
            languages_text = text
        
        # Check for languages in text
        for language in common_languages:
            if re.search(r'\b' + re.escape(language) + r'\b', languages_text):
                languages.append(language)
        
        return languages
    
    def _extract_section(self, text: str, section_names: List[str]) -> str:
        """Extract a specific section from CV text"""
        lines = text.split('\n')
        section_text = ""
        in_section = False
        
        for i, line in enumerate(lines):
            # Look for section header
            if not in_section:
                for name in section_names:
                    if re.search(r'\b' + re.escape(name) + r'\b', line, re.IGNORECASE):
                        in_section = True
                        break
            # If in section, add line to section text
            elif in_section:
                # Check if we've reached the next section
                if line.strip() and line.strip() == line.strip().upper() and len(line.strip()) > 3:
                    # Likely a new section header
                    break
                section_text += line + '\n'
        
        return section_text
    
    def _skill_match(self, required_skill: str, candidate_skills: List[str]) -> bool:
        """Check if a required skill matches any candidate skill"""
        for skill in candidate_skills:
            # Check for exact match
            if required_skill.lower() == skill.lower():
                return True
            
            # Check for partial match
            if required_skill.lower() in skill.lower() or skill.lower() in required_skill.lower():
                return True
                
        return False
    
    def _analyze_experience_match(self, experiences: List[Dict[str, str]], 
                                min_years: int) -> float:
        """Analyze if experience matches the requirements"""
        if not experiences:
            return 0.0
            
        # Try to extract years from experience
        total_years = 0
        for exp in experiences:
            duration = exp.get("duration", "")
            # Look for year patterns (e.g. 2018-2021, 2018 to Present)
            years_match = re.findall(r'(19|20)\d{2}', duration)
            if len(years_match) >= 2:
                try:
                    start_year = int(years_match[0])
                    
                    if "present" in duration.lower():
                        import datetime
                        end_year = datetime.datetime.now().year
                    else:
                        end_year = int(years_match[1])
                        
                    years = end_year - start_year
                    if 0 <= years <= 30:  # Sanity check
                        total_years += years
                except:
                    pass
        
        # Calculate match score
        if total_years >= min_years:
            return 1.0
        elif total_years > 0:
            return total_years / min_years
        else:
            return 0.1  # Give some small score even if years couldn't be determined
    
    def _analyze_education_match(self, education: List[Dict[str, str]], 
                            requirements: Dict[str, Any]) -> float:
        """Analyze if education matches requirements"""
        if not education or not requirements:
            return 0.5  # Neutral score if can't determine
            
        req_level = requirements.get("level", "").lower()
        
        # Check for degree level
        has_matching_degree = False
        for edu in education:
            degree = edu.get("degree", "").lower()
            
            if req_level == "bachelor" and any(term in degree for term in ["bachelor", "b.s.", "b.a."]):
                has_matching_degree = True
                break
            elif req_level == "master" and any(term in degree for term in ["master", "m.s.", "m.a."]):
                has_matching_degree = True
                break
            elif req_level == "phd" and any(term in degree for term in ["phd", "ph.d.", "doctorate"]):
                has_matching_degree = True
                break
        
        # Check for specific field if required
        req_field = requirements.get("field", "").lower()
        field_match = False
        
        if req_field and has_matching_degree:
            for edu in education:
                if req_field in edu.get("degree", "").lower():
                    field_match = True
                    break
        
        # Calculate match score
        if not req_level:
            return 0.8  # No specific requirements
        if has_matching_degree and (not req_field or field_match):
            return 1.0
        elif has_matching_degree:
            return 0.7  # Right level but wrong field
        else:
            # Check if they have higher education than required
            for edu in education:
                degree = edu.get("degree", "").lower()
                if req_level == "bachelor" and any(term in degree for term in ["master", "phd", "doctorate"]):
                    return 0.8  # Higher education than required
            
            return 0.3  # No matching degree