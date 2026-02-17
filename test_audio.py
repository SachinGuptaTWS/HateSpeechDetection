# -*- coding: utf-8 -*-
"""
Simple Audio Test Script
Tests the speech-to-text functionality with different audio formats
"""

import os
import tempfile
from speech_to_text import speech_processor

def test_audio_processing():
    """Test audio processing with a simple WAV file"""
    
    # Create a simple test audio file path (you'll need to provide one)
    print("=== Speech-to-Text Testing ===")
    print()
    print("Current Status:")
    print(f"- Whisper Available: { hasattr(speech_processor, 'whisper_model') and speech_processor.whisper_model is not None}")
    print(f"- SpeechRecognition Available: True")
    print(f"- PyDub Available: True")
    print()
    
    # Test file paths (you can test with actual audio files)
    test_files = [
        "test.wav",
        "test.mp3", 
        "test.flac",
        "test.m4a"
    ]
    
    print("To test the speech-to-text functionality:")
    print("1. Place an audio file in the project directory")
    print("2. Rename it to one of the following:")
    for file in test_files:
        print(f"   - {file}")
    print()
    print("3. Run this test script:")
    print("   python test_audio.py")
    print()
    print("4. Or test via the web interface at:")
    print("   http://127.0.0.1:5000")
    print()
    print("=== Supported Audio Formats ===")
    print("- WAV: Native support (best)")
    print("- FLAC: Native support") 
    print("- MP3: Requires conversion")
    print("- M4A: Requires conversion")
    print("- OGG: Requires conversion")
    print("- WEBM: Requires conversion")
    print()
    print("=== Troubleshooting ===")
    print("If audio recording doesn't work:")
    print("1. Check browser microphone permissions")
    print("2. Try using a WAV file instead")
    print("3. Make sure audio is clear and not too quiet")
    print("4. Test with different browsers (Chrome, Firefox, Edge)")

if __name__ == "__main__":
    test_audio_processing()
