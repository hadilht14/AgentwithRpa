"""
Audio Extraction Module - Transcribes and processes speech from audio files
Uses SpeechRecognition and PyDub for audio processing and transcription
"""
import logging
import time
import os
import json
from typing import Dict, Any, List, Optional, Union
import io

# Import necessary libraries
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Configure logging
logger = logging.getLogger(__name__)

class AudioExtractor:
    def __init__(self, db_instance):
        """Initialize with a database connection"""
        self.db = db_instance
        self.recognizer = sr.Recognizer()
        self.transcriptions = {}  # Cache for transcriptions
        logger.info("Audio Extractor initialized")
    
    def transcribe_audio_file(self, file_path: str, language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe speech from an audio file
        
        Args:
            file_path: Path to the audio file
            language: Language code (default: en-US)
            
        Returns:
            Dictionary with transcription and metadata
        """
        logger.info(f"Transcribing audio file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found", "text": ""}
        
        try:
            # Determine file type by extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Process based on file type
            if file_extension == '.wav':
                return self._process_wav(file_path, language)
            elif file_extension in ['.mp3', '.ogg', '.flac', '.aac', '.m4a']:
                return self._convert_and_process(file_path, file_extension, language)
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return {"error": f"Unsupported file format: {file_extension}", "text": ""}
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"error": f"Error transcribing audio: {str(e)}", "text": ""}
    
    def transcribe_audio_bytes(self, audio_bytes: bytes, format_hint: str = "wav",language: str = "en-US") -> Dict[str, Any]:
        """
        Transcribe speech from audio bytes (for uploaded files)
        
        Args:
            audio_bytes: Audio file content as bytes
            format_hint: Audio format (wav, mp3, etc.)
            language: Language code (default: en-US)
            
        Returns:
            Dictionary with transcription and metadata
        """
        logger.info(f"Transcribing audio bytes (format: {format_hint})")
        
        try:
            # Create a temporary in-memory file
            audio_io = io.BytesIO(audio_bytes)
            
            # Process based on format hint
            if format_hint.lower() == 'wav':
                # Direct processing for WAV
                with sr.AudioFile(audio_io) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data, language=language)
                    
                    result = {
                        "text": text,
                        "duration": None,  # Can't determine without processing the audio
                        "language": language,
                        "timestamp": time.time()
                    }
                    
                    # Generate a unique ID for this transcription
                    import uuid
                    trans_id = f"trans_{str(uuid.uuid4())[:8]}"
                    self.transcriptions[trans_id] = result
                    
                    # Log the transcription
                    self.db["agent_log"].append({
                        "action": "transcribe_audio_bytes", 
                        "params": {"format": format_hint, "language": language}, 
                        "timestamp": time.time()
                    })
                    
                    return result
            else:
                # For other formats, convert to WAV first
                audio = AudioSegment.from_file(audio_io, format=format_hint)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                
                with sr.AudioFile(wav_io) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data, language=language)
                    
                    result = {
                        "text": text,
                        "duration": len(audio) / 1000.0,  # Convert ms to seconds
                        "language": language,
                        "timestamp": time.time()
                    }
                    
                    # Generate a unique ID for this transcription
                    import uuid
                    trans_id = f"trans_{str(uuid.uuid4())[:8]}"
                    self.transcriptions[trans_id] = result
                    
                    # Log the transcription
                    self.db["agent_log"].append({
                        "action": "transcribe_audio_bytes", 
                        "params": {"format": format_hint, "language": language}, 
                        "timestamp": time.time()
                    })
                    
                    return result
                
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return {"error": "Speech recognition could not understand audio", "text": ""}
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {str(e)}")
            return {"error": f"Speech recognition service error: {str(e)}", "text": ""}
        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {str(e)}")
            return {"error": f"Error transcribing audio bytes: {str(e)}", "text": ""}
    
    def process_hr_audio(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process HR-related audio (interviews, meetings, etc.)
        
        Args:
            audio_data: Dictionary with transcription data
            
        Returns:
            Dictionary with processed information and action items
        """
        logger.info("Processing HR audio content")
        
        transcription = audio_data.get("text", "")
        if not transcription:
            return {"error": "No transcription text to process"}
        
        # Split into sentences for analysis
        import nltk # type: ignore
        try:
            sentences = nltk.sent_tokenize(transcription)
        except:
            # Fallback if NLTK not fully installed
            sentences = transcription.split('.')
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Look for key HR topics and action items
        hr_keywords = {
            "leave": ["vacation", "leave", "time off", "sick", "absence"],
            "performance": ["performance", "review", "evaluation", "feedback", "improvement"],
            "complaint": ["issue", "concern", "problem", "complaint", "harassment"],
            "hiring": ["candidate", "interview", "hiring", "recruitment", "position"],
            "termination": ["terminate", "fire", "let go", "severance", "resignation"]
        }
        
        # Find matching topics
        topics_found = {}
        for topic, keywords in hr_keywords.items():
            matches = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    matches.append(sentence)
            if matches:
                topics_found[topic] = matches
        
        # Look for action items (sentences with action verbs followed by "will" or similar)
        action_verbs = ["schedule", "complete", "review", "submit", "approve", "prepare", "contact"]
        action_items = []
        
        for sentence in sentences:
            s_lower = sentence.lower()
            if any(f" will {verb}" in s_lower or f" to {verb}" in s_lower for verb in action_verbs):
                action_items.append(sentence)
        
        # Extract potential dates (simple pattern matching)
        import re
        date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2}(st|nd|rd|th)?(,?\s+\d{4})?\b'
        dates = []
        
        for sentence in sentences:
            date_matches = re.findall(date_pattern, sentence, re.IGNORECASE)
            if date_matches:
                for match in date_matches:
                    # Extract the date with surrounding context
                    start_idx = sentence.lower().find(match[0].lower())
                    if start_idx >= 0:
                        context_start = max(0, start_idx - 30)
                        context_end = min(len(sentence), start_idx + len(match[0]) + 50)
                        date_context = sentence[context_start:context_end].strip()
                        dates.append(date_context)
        
        # Create summary
        summary = {
            "topics_identified": list(topics_found.keys()),
            "topic_sentences": topics_found,
            "action_items": action_items,
            "dates_mentioned": dates,
            "transcript_length": len(transcription),
            "sentence_count": len(sentences),
            "timestamp": time.time()
        }
        
        # Log the processing
        self.db["agent_log"].append({
            "action": "process_hr_audio", 
            "params": {"transcript_length": len(transcription)}, 
            "timestamp": time.time()
        })
        
        return summary
    
    def create_audio_decision(self, audio_data: Dict[str, Any], processing_result: Dict[str, Any]) -> str:
        """
        Create an automated decision record for audio content
        
        Args:
            audio_data: Original transcription data
            processing_result: Processed information
            
        Returns:
            Decision ID
        """
        logger.info("Creating automated audio processing decision")
        
        # Determine if this needs HR attention based on topics
        topics = processing_result.get("topics_identified", [])
        needs_attention = any(topic in ["complaint", "termination"] for topic in topics)
        
        # Set confidence based on identified topics and actions
        confidence_score = 0.5  # Default moderate confidence
        if needs_attention:
            confidence_score = 0.3  # Lower confidence for sensitive topics
        elif len(processing_result.get("action_items", [])) > 0:
            confidence_score = 0.7  # Higher confidence when clear actions found
        
        # Generate justification
        justification = f"Audio contains topics: {', '.join(topics)}. "
        if needs_attention:
            justification += "Contains sensitive topics requiring human review. "
        
        if processing_result.get("action_items", []):
            justification += f"Contains {len(processing_result['action_items'])} actionable items. "
        
        if processing_result.get("dates_mentioned", []):
            justification += f"Contains {len(processing_result['dates_mentioned'])} date references."
        
        # Create decision parameters
        parameters = {
            "transcript_snippet": audio_data.get("text", "")[:200] + "...",  # First 200 chars
            "topics": topics,
            "needs_immediate_attention": needs_attention,
            "action_item_count": len(processing_result.get("action_items", [])),
            "audio_duration": audio_data.get("duration", "unknown")
        }
        
        # Use the existing API to create a decision
        from main import create_ai_decision_impl
        
        decision_result = create_ai_decision_impl(
            action_name="process_hr_audio_recording",
            parameters=parameters,
            confidence_score=confidence_score,
            justification=justification
        )
        
        # Log the decision creation
        self.db["agent_log"].append({
            "action": "create_audio_decision", 
            "params": {"topics": topics}, 
            "timestamp": time.time()
        })
        
        return decision_result
    
    def extract_interview_evaluation(self, transcription: str) -> Dict[str, Any]:
        """
        Extract interview evaluation metrics from a job interview transcription
        
        Args:
            transcription: Interview transcription text
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Extracting interview evaluation metrics")
        
        # Split into interviewer and candidate parts (simplistic approach)
        lines = transcription.split('\n')
        interviewer_text = ""
        candidate_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Interviewer:") or line.startswith("Q:"):
                interviewer_text += line.split(":", 1)[1].strip() + " "
            elif line.startswith("Candidate:") or line.startswith("A:"):
                candidate_text += line.split(":", 1)[1].strip() + " "
        
        # For more sophisticated analysis, we would use NLP here
        # This is a simplified version
        
        # Look for skill keywords
        skill_keywords = [
            "experience", "project", "develop", "team", "manage", "lead",
            "problem", "solution", "implement", "design", "communicate"
        ]
        
        skills_mentioned = []
        for skill in skill_keywords:
            if skill in candidate_text.lower():
                skills_mentioned.append(skill)
        
        # Simple sentiment analysis
        positive_words = ["success", "achieve", "improve", "excellent", "great", "good", "effective"]
        negative_words = ["fail", "issue", "problem", "difficult", "bad", "struggle"]
        
        positive_count = sum(candidate_text.lower().count(word) for word in positive_words)
        negative_count = sum(candidate_text.lower().count(word) for word in negative_words)
        
        sentiment_score = positive_count - negative_count
        
        # Response length analysis
        avg_response_length = len(candidate_text) / max(candidate_text.count(". "), 1)
        
        # Create evaluation
        evaluation = {
            "skills_mentioned": skills_mentioned,
            "communication_metrics": {
                "average_response_length": avg_response_length,
                "sentiment_score": sentiment_score
            },
            "overall_impression": "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral",
            "timestamp": time.time()
        }
        
        # Log the evaluation
        self.db["agent_log"].append({
            "action": "extract_interview_evaluation", 
            "params": {"text_length": len(transcription)}, 
            "timestamp": time.time()
        })
        
        return evaluation
    
    def _process_wav(self, file_path: str, language: str) -> Dict[str, Any]:
        """Process WAV file and transcribe"""
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=language)
                
                # Get duration if possible
                duration = None
                try:
                    audio = AudioSegment.from_wav(file_path)
                    duration = len(audio) / 1000.0  # Convert ms to seconds
                except:
                    logger.warning("Could not determine audio duration")
                
                result = {
                    "text": text,
                    "duration": duration,
                    "language": language,
                    "timestamp": time.time()
                }
                
                # Cache the result
                trans_id = os.path.basename(file_path)
                self.transcriptions[trans_id] = result
                
                # Log the transcription
                self.db["agent_log"].append({
                    "action": "transcribe_wav_file", 
                    "params": {"file_path": file_path, "language": language}, 
                    "timestamp": time.time()
                })
                
                return result
                
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return {"error": "Speech recognition could not understand audio", "text": ""}
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {str(e)}")
            return {"error": f"Speech recognition service error: {str(e)}", "text": ""}
    
    def _convert_and_process(self, file_path: str, file_extension: str, language: str) -> Dict[str, Any]:
        """Convert non-WAV audio to WAV and process"""
        try:
            # Determine format from extension without the dot
            format_name = file_extension[1:]
            
            # Load audio file
            audio = AudioSegment.from_file(file_path, format=format_name)
            
            # Create a temporary WAV file
            temp_wav = os.path.join(os.path.dirname(file_path), "temp_convert.wav")
            audio.export(temp_wav, format="wav")
            
            # Process the WAV file
            result = self._process_wav(temp_wav, language)
            
            # Add original format info
            result["original_format"] = format_name
            
            # Clean up temp file
            try:
                os.remove(temp_wav)
            except:
                logger.warning(f"Could not remove temporary file: {temp_wav}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            return {"error": f"Error converting audio: {str(e)}", "text": ""}
            
    def transcribe_long_audio(self, file_path: str, language: str = "en-US",chunk_size_ms: int = 60000) -> Dict[str, Any]:
        """
        Transcribe long audio file by breaking it into chunks
        
        Args:
            file_path: Path to the audio file
            language: Language code (default: en-US)
            chunk_size_ms: Size of each chunk in milliseconds (default: 60000 = 1 minute)
            
        Returns:
            Dictionary with full transcription and metadata
        """
        logger.info(f"Transcribing long audio file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": "File not found", "text": ""}
        
        try:
            # Determine file type by extension
            file_extension = os.path.splitext(file_path)[1].lower()
            format_name = file_extension[1:]  # Remove the dot
            
            # Load audio file
            audio = AudioSegment.from_file(file_path, format=format_name)
            
            # Get total duration
            duration = len(audio) / 1000.0  # Convert ms to seconds
            
            # Split audio into chunks
            chunks = []
            for i in range(0, len(audio), chunk_size_ms):
                chunks.append(audio[i:i+chunk_size_ms])
            
            logger.info(f"Split audio into {len(chunks)} chunks")
            
            # Process each chunk
            transcriptions = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Export chunk to WAV
                temp_wav = os.path.join(os.path.dirname(file_path), f"temp_chunk_{i}.wav")
                chunk.export(temp_wav, format="wav")
                
                # Transcribe chunk
                try:
                    chunk_result = self._process_wav(temp_wav, language)
                    if "text" in chunk_result and chunk_result["text"]:
                        transcriptions.append(chunk_result["text"])
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                
                # Clean up temp file
                try:
                    os.remove(temp_wav)
                except:
                    logger.warning(f"Could not remove temporary file: {temp_wav}")
            
            # Combine all transcriptions
            full_text = " ".join(transcriptions)
            
            result = {
                "text": full_text,
                "duration": duration,
                "language": language,
                "chunks_processed": len(chunks),
                "timestamp": time.time()
            }
            
            # Cache the result
            trans_id = os.path.basename(file_path)
            self.transcriptions[trans_id] = result
            
            # Log the transcription
            self.db["agent_log"].append({
                "action": "transcribe_long_audio", 
                "params": {"file_path": file_path, "chunks": len(chunks)}, 
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing long audio: {str(e)}")
            return {"error": f"Error transcribing long audio: {str(e)}", "text": ""}