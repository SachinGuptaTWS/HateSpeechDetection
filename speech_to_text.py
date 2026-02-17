# -*- coding: utf-8 -*-
"""
Speech-to-Text Module for Hate Speech Detection
Supports both offline Whisper and browser Web Speech API
"""

import os
import tempfile
import logging
from typing import Optional
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("OpenAI Whisper not available, will use SpeechRecognition only")
import speech_recognition as sr
from flask import request, jsonify
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, audio format conversion limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextProcessor:
    """Handles speech-to-text conversion using multiple methods"""
    
    def __init__(self):
        self.whisper_model = None
        self.recognizer = sr.Recognizer()
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper model (primary transcription method)"""
        if not WHISPER_AVAILABLE:
            logger.warning("Whisper not available, will use SpeechRecognition only")
            return
            
        try:
            # Check if CUDA is available for faster processing
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Initializing Whisper on device: {device}")
            except ImportError:
                device = "cpu"
                logger.info("PyTorch not available, using CPU for Whisper")
            
            # Load base model (good balance of speed and accuracy)
            logger.info("Loading Whisper base model...")
            self.whisper_model = whisper.load_model("base", device=device)
            logger.info("✅ Whisper model loaded successfully - PRIMARY METHOD READY")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
            self.whisper_model = None
    
    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        """Normalize language input to formats expected by backends."""
        if not language:
            return None

        lang = str(language).strip()
        if not lang:
            return None

        lower = lang.lower()
        if lower in {"en", "en-in", "en_us", "en-us"}:
            return "en"
        if lower in {"hi", "hi-in", "hi_in"}:
            return "hi"

        # Whisper accepts ISO 639-1 or some locale-like values; keep as-is.
        return lang

    def _normalize_google_language(self, language: Optional[str]) -> str:
        """Return language code for Google SpeechRecognition."""
        if not language:
            return "en-IN"

        lower = str(language).strip().lower()
        if lower in {"hi", "hi-in", "hi_in"}:
            return "hi-IN"
        if lower in {"en", "en-in", "en_us", "en-us"}:
            return "en-IN"

        # If user passed a locale already, try it.
        return str(language)

    def transcribe_with_whisper(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper (offline)
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available")
            return None
            
        if self.whisper_model is None:
            logger.error("Whisper model not initialized")
            return None
        
        try:
            logger.info(f"Transcribing {audio_file_path} with Whisper...")

            whisper_lang = self._normalize_language(language)

            # Transcribe the audio
            if whisper_lang:
                result = self.whisper_model.transcribe(audio_file_path, language=whisper_lang)
            else:
                result = self.whisper_model.transcribe(audio_file_path)
            transcribed_text = result["text"].strip()
            
            logger.info(f"Whisper transcription successful: '{transcribed_text[:50]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return None
    
    def convert_to_wav(self, audio_file_path: str) -> Optional[str]:
        """
        Convert audio file to WAV format for SpeechRecognition compatibility
        Simplified version that works without FFmpeg
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Path to converted WAV file or original path if conversion fails
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available, using original file")
            return audio_file_path
        
        try:
            # Check if already WAV
            if audio_file_path.lower().endswith('.wav'):
                logger.info(f"File {audio_file_path} is already WAV format")
                return audio_file_path
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file does not exist: {audio_file_path}")
                return None
            
            logger.info(f"Attempting to convert {audio_file_path} to WAV format...")
            
            # Try to convert without FFmpeg-specific parameters
            try:
                audio = AudioSegment.from_file(audio_file_path)
                wav_path = audio_file_path.rsplit('.', 1)[0] + '_converted.wav'
                
                # Simple export without FFmpeg parameters
                audio.export(wav_path, format='wav')
                
                # Verify the converted file exists and is readable
                if os.path.exists(wav_path):
                    # Test if the file can be read by SpeechRecognition
                    try:
                        with sr.AudioFile(wav_path) as test_source:
                            pass  # Just test if it opens
                        logger.info(f"Successfully converted {audio_file_path} to {wav_path}")
                        return wav_path
                    except Exception as test_e:
                        logger.warning(f"Converted file not readable: {test_e}")
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                        return None
                else:
                    logger.error(f"Failed to create WAV file: {wav_path}")
                    return None
                    
            except Exception as conversion_error:
                logger.error(f"Audio conversion failed: {conversion_error}")
                # Return original file if it's a supported format
                if audio_file_path.lower().endswith(('.wav', '.flac')):
                    logger.info(f"Using original file as fallback: {audio_file_path}")
                    return audio_file_path
                return None
            
        except Exception as e:
            logger.error(f"Failed to convert audio to WAV: {e}")
            return audio_file_path if os.path.exists(audio_file_path) else None
    
    def transcribe_with_speech_recognition(self, audio_file_path: str, language: Optional[str] = None) -> Optional[str]:
        """
        Transcribe audio using SpeechRecognition library (Google's free API)
        Enhanced with validation to prevent mock results
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if failed
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file does not exist: {audio_file_path}")
            return None
            
        # Validate file size and format
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:  # Less than 1KB is likely invalid
            logger.error(f"Audio file too small: {file_size} bytes")
            return None
            
        wav_path = None
        try:
            logger.info(f"Transcribing {audio_file_path} with SpeechRecognition...")
            logger.info(f"File size: {file_size} bytes")
            
            # Convert to WAV format if needed
            wav_path = self.convert_to_wav(audio_file_path)
            
            if not wav_path or not os.path.exists(wav_path):
                logger.error(f"Failed to convert audio to WAV: {audio_file_path}")
                return None
            
            # Validate converted file
            converted_size = os.path.getsize(wav_path)
            if converted_size < 1000:
                logger.error(f"Converted WAV file too small: {converted_size} bytes")
                return None
            
            # Try to read the audio file first
            with sr.AudioFile(wav_path) as source:
                logger.info(f"Audio file loaded successfully: {wav_path}")
                logger.info(f"Audio duration: {source.DURATION} seconds")
                
                # Validate audio duration
                if source.DURATION < 0.1:  # Less than 0.1 seconds is invalid
                    logger.error(f"Audio duration too short: {source.DURATION} seconds")
                    return None
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
                
                # Validate audio data
                if not audio_data or len(audio_data.get_raw_data()) < 1000:
                    logger.error("Audio data is empty or too small")
                    return None
                
                # Use Google's free speech recognition (no API key needed)
                google_lang = self._normalize_google_language(language)
                logger.info(f"Sending audio to Google Speech Recognition API (language={google_lang})...")
                text = self.recognizer.recognize_google(audio_data, language=google_lang)
                
                # Validate transcription result
                if not text:
                    logger.error("Empty transcription result")
                    return None
                
                if len(text.strip()) < 2:
                    logger.warning(f"Very short transcription: '{text}'")
                    # Still accept short results if they're meaningful
                
                # Check for common error responses
                error_patterns = [
                    "could not understand",
                    "could not transcribe", 
                    "audio not clear",
                    "try again",
                    "error",
                    "unknown"
                ]
                
                text_lower = text.lower()
                for pattern in error_patterns:
                    if pattern in text_lower:
                        logger.error(f"Likely error response detected: '{text}'")
                        return None
                
                # Log success with confidence indicators
                logger.info(f"SpeechRecognition transcription successful")
                logger.info(f"Transcribed text: '{text}'")
                logger.info(f"Text length: {len(text)} characters")
                logger.info(f"Word count: {len(text.split())} words")
                
                return text
                
        except sr.UnknownValueError:
            logger.error("SpeechRecognition could not understand the audio")
            return None
        except sr.RequestError as e:
            logger.error(f"SpeechRecognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"SpeechRecognition transcription failed: {e}")
            return None
        finally:
            # Clean up converted WAV file if different from original
            if wav_path and wav_path != audio_file_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                    logger.info(f"Cleaned up temporary WAV file: {wav_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {wav_path}: {e}")
    
    def transcribe_audio(self, audio_file_path: str, method: str = "whisper", language: Optional[str] = None) -> Optional[str]:
        """
        Main method to transcribe audio file
        Whisper is prioritized as the primary transcription method
        
        Args:
            audio_file_path: Path to audio file
            method: "whisper" (primary) or "speech_recognition" (fallback)
            
        Returns:
            Transcribed text or None if failed
        """
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
        
        if method == "whisper":
            if WHISPER_AVAILABLE and self.whisper_model is not None:
                logger.info("🎯 Using WHISPER (Primary Method)")
                result = self.transcribe_with_whisper(audio_file_path, language=language)
                if result:
                    logger.info("✅ Whisper transcription successful")
                    return result
                else:
                    logger.warning("⚠️ Whisper failed, falling back to SpeechRecognition")
                    return self.transcribe_with_speech_recognition(audio_file_path, language=language)
            else:
                logger.warning("⚠️ Whisper not available, using SpeechRecognition")
                return self.transcribe_with_speech_recognition(audio_file_path, language=language)
        elif method == "speech_recognition":
            logger.info("🎤 Using SpeechRecognition (Requested Method)")
            return self.transcribe_with_speech_recognition(audio_file_path, language=language)
        else:
            logger.error(f"❌ Unknown transcription method: {method}")
            return None
    
    def save_uploaded_audio(self, audio_file) -> Optional[str]:
        """
        Save uploaded audio file to temporary location
        
        Args:
            audio_file: File object from Flask request
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create temporary directory if it doesn't exist
            temp_dir = tempfile.gettempdir()
            
            # Generate unique filename
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            
            # Get original filename and extension
            if hasattr(audio_file, 'filename'):
                original_filename = audio_file.filename
                if '.' in original_filename:
                    file_ext = original_filename.rsplit('.', 1)[1].lower()
                else:
                    file_ext = 'wav'  # Default to wav
            else:
                file_ext = 'wav'
                original_filename = f'recording_{unique_id}.wav'
            
            # Create filename
            filename = f'audio_{unique_id}.{file_ext}'
            file_path = os.path.join(temp_dir, filename)
            
            logger.info(f"Saving audio file: {original_filename} -> {file_path}")
            
            # Save file
            audio_file.save(file_path)
            
            # Verify file was saved and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                logger.info(f"Audio file saved successfully: {file_path} ({os.path.getsize(file_path)} bytes)")
                return file_path
            else:
                logger.error(f"Failed to save audio file or file is empty: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save uploaded audio: {e}")
            return None
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary audio file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")

# Global instance
speech_processor = SpeechToTextProcessor()
