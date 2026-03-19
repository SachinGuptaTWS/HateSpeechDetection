# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 08:39:20 2023

@author: Sachin Gupta
"""

from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import tensorflow as tf
import tensorflow_text  # prerequisite for using the BERT preprocessing layer
import numpy as np
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from speech_to_text import speech_processor
from data_ingestion import data_manager

# Load environment variables from .env file FIRST
load_dotenv()

# We'll check availability at runtime instead of import time
GEMINI_AVAILABLE = None
TRANSLATOR_AVAILABLE = None

# Create the Flask web application
app = Flask(__name__)

# Set a secret key (stored in .env) as a security measure (e.g. protecting against CSRF attacks) 
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

# Configure file upload settings
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "m4a", "ogg", "webm"}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the TensorFlow model
model = tf.keras.models.load_model("saved_models/model3")


def debug_gemini_models():
    """Debug function to list available Gemini models"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        app.logger.warning("No GEMINI_API_KEY found for debug")
        return
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        app.logger.info("🔍 Available Gemini models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                app.logger.info(f"  - {m.name}")
    except Exception as e:
        app.logger.error(f"❌ Failed to list models: {e}")


def gemini_hate_speech_detection(text: str) -> tuple[str, float]:
    """Primary hate speech detection using Gemini API"""
    if not text:
        return "No Hate Speech", 0.0
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    app.logger.info(f"🔍 AI Service debug: API key present = {bool(gemini_key)}")
    
    # Check AI service availability at runtime
    global GEMINI_AVAILABLE
    if GEMINI_AVAILABLE is None:
        try:
            import google.generativeai as genai
            GEMINI_AVAILABLE = True
            app.logger.info("✅ AI service library imported successfully")
        except Exception as e:
            GEMINI_AVAILABLE = False
            app.logger.error(f"❌ AI service library import failed: {e}")
    
    if gemini_key and GEMINI_AVAILABLE:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model_g = genai.GenerativeModel("gemini-2.5-flash")
            
            prompt = (
                "You are an expert content moderation AI specializing in hate speech and sarcasm detection. "
                "Analyze the following text for hate speech, considering context, nuance, and potential sarcasm.\n\n"
                "Evaluation criteria:\n"
                "- Hate Speech: Content that attacks, threatens, or degrades individuals/groups based on protected characteristics (race, religion, ethnicity, gender, sexual orientation, disability, etc.)\n"
                "- Sarcasm: Statements that mean the opposite of what they literally express, often used to mock or convey contempt\n"
                "- Context matters: Consider whether seemingly offensive content might be sarcasm, irony, or legitimate criticism\n\n"
                "Respond with ONLY 'Hate Speech' or 'No Hate Speech' followed by a confidence score between 0.0 and 1.0. "
                "Be conservative - if uncertain, default to 'No Hate Speech' with lower confidence.\n\n"
                "Format: 'RESULT|CONFIDENCE'\n\n"
                f"Text: {text}"
            )
            
            app.logger.info(f"🤖 Calling AI service for text: '{text[:50]}...'")
            response = model_g.generate_content(prompt)
            
            # Check for safety filter blocks
            if response.candidates:
                result = response.text.strip()
                app.logger.info(f"📝 AI service raw response: '[REDACTED]'")
                if '|' in result:
                    prediction, confidence = result.split('|', 1)
                    prediction = prediction.strip()
                    try:
                        confidence = float(confidence.strip())
                        app.logger.info(f"✅ AI service detection: {prediction} ({confidence:.2f})")
                        return prediction, confidence
                    except ValueError:
                        app.logger.warning(f"⚠️ AI service confidence parsing failed")
                        pass
                
                # Fallback if format is unexpected - be more conservative
                result_lower = result.lower()
                hate_indicators = ['hate', 'kill', 'die', 'attack', 'violent', 'threat']
                sarcasm_indicators = ['sarcasm', 'irony', 'joke', 'kidding', 'obviously', 'clearly']
                
                # Check for sarcasm indicators
                if any(indicator in result_lower for indicator in sarcasm_indicators):
                    return "No Hate Speech", 0.3
                # Check for strong hate indicators
                elif any(indicator in result_lower for indicator in hate_indicators):
                    return "Hate Speech", 0.7
                # Default to conservative approach
                else:
                    return "No Hate Speech", 0.4
            else:
                app.logger.warning("⚠️ AI service response blocked by content filters")
                return None, None
                    
        except Exception as e:
            app.logger.error(f"❌ AI service detection failed: {e}")
    else:
        app.logger.warning(f"⚠️ AI service not available: key={bool(gemini_key)}, available={GEMINI_AVAILABLE}")
    
    return None, None


def translate_hi_to_en(text: str) -> str | None:
    if not text:
        return None

    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Check Gemini availability at runtime
    global GEMINI_AVAILABLE
    if GEMINI_AVAILABLE is None:
        try:
            import google.generativeai as genai
            GEMINI_AVAILABLE = True
        except Exception:
            GEMINI_AVAILABLE = False
    
    if gemini_key and GEMINI_AVAILABLE:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model_g = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "Translate the following Hindi text to English. "
                "Return ONLY the English translation, no explanations.\n\n"
                f"Text: {text}"
            )
            response = model_g.generate_content(prompt)
            if response and response.text:
                translation = response.text.strip()
                app.logger.info(f"✅ Gemini translation successful: '{translation[:80]}...'")
                return translation
        except Exception as e:
            app.logger.warning(f"Gemini translation failed: {e}")
    
    # Check googletrans availability at runtime
    global TRANSLATOR_AVAILABLE
    if TRANSLATOR_AVAILABLE is None:
        try:
            from googletrans import Translator
            TRANSLATOR_AVAILABLE = True
        except Exception:
            TRANSLATOR_AVAILABLE = False
    
    if TRANSLATOR_AVAILABLE:
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, src='hi', dest='en')
            if result and result.text:
                translation = result.text.strip()
                app.logger.info(f"✅ googletrans translation successful: '{translation[:80]}...'")
                return translation
        except Exception as e:
            app.logger.warning(f"googletrans translation failed: {e}")
    
    app.logger.warning("Both Gemini and googletrans translation failed")
    return None


# Create hate speech detection form class (that inherits from the Flask WTForm class)
class HateSpeechForm(FlaskForm):
    comment = StringField("Social Media Comment", validators=[DataRequired()])
    submit = SubmitField("Run")


# Home route 
@app.route("/", methods=["GET", "POST"])
def home():
    # Instantiate a hate speech form class object
    form = HateSpeechForm()
    # If the user submitted valid information in the hate speech form
    if form.validate_on_submit():
        # Get the input text from the form
        input_text = form.comment.data
        
        # Try Gemini first (primary method)
        gemini_prediction, gemini_confidence = gemini_hate_speech_detection(input_text)
        
        if gemini_prediction is not None:
            prediction = gemini_prediction
            prediction_prob = gemini_confidence * 100  # Convert to percentage
        else:
            # Fallback to TensorFlow model
            input_data = [input_text]
            prediction_prob = model.predict(input_data)[0][0]
            prediction_prob = np.round(prediction_prob * 100, 1)
            if prediction_prob >= 50:
                prediction = "Hate Speech"
            else:
                prediction = "No Hate Speech"
                prediction_prob = 100 - prediction_prob
        # Render the prediction and prediction probability in the index.html template
        return render_template("index.html", 
                               form=form, 
                               prediction=prediction, 
                               prediction_prob=prediction_prob)
    return render_template("index.html", form=form)


# API route
@app.route("/api")
def prediction_by_api():
    # Get the input text from the api query parameter
    input_text = request.args.get("comment")
    
    # Try Gemini first (primary method)
    gemini_prediction, gemini_confidence = gemini_hate_speech_detection(input_text)
    
    if gemini_prediction is not None:
        prediction = gemini_prediction
        prediction_prob = gemini_confidence
    else:
        # Fallback to TensorFlow model
        input_data = [input_text]
        prediction_prob = model.predict(input_data)[0][0]
        if prediction_prob >= 0.5:
            prediction = "Hate Speech"
        else:
            prediction = "No Hate Speech"
            prediction_prob = 1 - prediction_prob
    # Return json with the prediction and prediction probability
    return jsonify({"prediction": prediction,
                    "probability": float(prediction_prob)})


# Text analysis API route
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    try:
        # Get the input text from the request
        input_text = request.json.get("text", "")
        
        # Try Gemini first (primary method)
        gemini_prediction, gemini_confidence = gemini_hate_speech_detection(input_text)
        
        if gemini_prediction is not None:
            prediction = gemini_prediction
            prediction_prob = gemini_confidence
            prediction_prob_percent = gemini_confidence * 100
        else:
            # Fallback to TensorFlow model
            input_data = [input_text]
            prediction_prob = model.predict(input_data)[0][0]
            prediction_prob_percent = np.round(prediction_prob * 100, 1)
            if prediction_prob >= 0.5:
                prediction = "Hate Speech"
            else:
                prediction = "No Hate Speech"
                prediction_prob = 1 - prediction_prob
                prediction_prob_percent = np.round(prediction_prob * 100, 1)
        
        # Return json with the prediction and prediction probability
        return jsonify({
            "prediction": prediction,
            "probability": prediction_prob_percent / 100,  # Convert back to decimal
            "text": input_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Speech-to-Text upload route
@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    """Handle audio file upload and transcription with validation"""
    
    # Check if file was uploaded
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files["audio"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Allowed types: wav, mp3, flac, m4a, ogg, webm"}), 400
    
    try:
        # Validate file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if file_size < 1000:  # Less than 1KB
            return jsonify({"error": "Audio file too small or empty"}), 400
        
        # Save uploaded audio file
        audio_path = speech_processor.save_uploaded_audio(file)
        
        if not audio_path:
            return jsonify({"error": "Failed to save audio file"}), 500
        
        # Transcribe audio with validation
        method = request.form.get("method", "whisper")
        language = request.form.get("language")
        transcribed_text = speech_processor.transcribe_audio(audio_path, method, language=language)
        
        # Clean up temporary file
        speech_processor.cleanup_temp_file(audio_path)
        
        # Validate transcription result
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio - speech not recognized"}), 500
        
        # Additional validation for transcription quality
        transcribed_text = transcribed_text.strip()
        if len(transcribed_text) < 2:
            return jsonify({"error": "Transcription too short or unclear - please speak clearly"}), 400
        
        # Check for potential error responses
        error_indicators = [
            "could not understand", "could not transcribe", "audio not clear",
            "try again", "error", "unknown", "transcription failed"
        ]
        
        text_lower = transcribed_text.lower()
        for indicator in error_indicators:
            if indicator in text_lower:
                return jsonify({"error": f"Audio transcription failed: {transcribed_text}"}), 500
        
        # Validate that transcription contains meaningful content
        if not any(c.isalpha() for c in transcribed_text):
            return jsonify({"error": "Transcription contains no recognizable text"}), 500
        
        # Translate Hindi->English before classification (when language is Hindi)
        classification_text = transcribed_text
        translated_text = None
        if language and language.lower().startswith("hi"):
            gemini_key = os.getenv("GEMINI_API_KEY")
            # Force runtime availability check
            global GEMINI_AVAILABLE, TRANSLATOR_AVAILABLE
            
            # Check Gemini availability
            if GEMINI_AVAILABLE is None:
                try:
                    import google.generativeai
                    GEMINI_AVAILABLE = True
                except Exception as e:
                    GEMINI_AVAILABLE = False
                    app.logger.debug(f"Gemini import failed: {e}")
            
            # Check googletrans availability  
            if TRANSLATOR_AVAILABLE is None:
                try:
                    from googletrans import Translator
                    TRANSLATOR_AVAILABLE = True
                except Exception as e:
                    TRANSLATOR_AVAILABLE = False
                    app.logger.debug(f"googletrans import failed: {e}")
            
            app.logger.info(f"Translation debug: GEMINI_AVAILABLE={GEMINI_AVAILABLE}, TRANSLATOR_AVAILABLE={TRANSLATOR_AVAILABLE}, GEMINI_API_KEY set={bool(gemini_key)}")
            
            if not gemini_key and not TRANSLATOR_AVAILABLE:
                app.logger.warning("Hindi selected but no translator available (set GEMINI_API_KEY or install googletrans)")
            
            translated_text = translate_hi_to_en(transcribed_text)
            if translated_text:
                classification_text = translated_text
                app.logger.info(f"Translation applied (hi->en): '{translated_text[:80]}...'")
            else:
                app.logger.warning("Hindi selected but translation failed; classifying original Hindi text")

        # Try Gemini first for hate speech detection
        gemini_prediction, gemini_confidence = gemini_hate_speech_detection(classification_text)
        
        if gemini_prediction is not None:
            prediction = gemini_prediction
            prediction_prob = gemini_confidence
        else:
            # Fallback to TensorFlow model
            input_data = [classification_text]
            prediction_prob = model.predict(input_data)[0][0]
            
            # Convert prediction probability to prediction in text form
            if prediction_prob >= 0.5:
                prediction = "Hate Speech"
            else:
                prediction = "No Hate Speech"
                prediction_prob = 1 - prediction_prob
        
        # Log successful processing for debugging
        app.logger.info(f"Successfully processed audio: '{classification_text[:50]}...'")
        app.logger.info(f"Prediction: {prediction} ({prediction_prob:.2f})")
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "translated_text": translated_text,
            "classified_text": classification_text,
            "prediction": prediction,
            "probability": float(prediction_prob),
            "method_used": method,
            "language_used": language
        })
        
    except Exception as e:
        app.logger.error(f"Audio processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# Data Ingestion Routes

@app.route("/ingest/twitter", methods=["POST"])
def ingest_twitter():
    """Ingest data from Twitter API"""
    try:
        data = request.json
        query = data.get("query", "")
        count = data.get("count", 100)
        language = data.get("language", "en")
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Ingest data
        ingested_data = data_manager.ingest_from_twitter(query, count, language)
        
        if not ingested_data:
            return jsonify({"error": "No data ingested from Twitter"}), 404
        
        # Process with hate speech detection
        processed_data = data_manager.process_ingested_data(ingested_data, model)
        
        return jsonify({
            "status": "success",
            "source": "twitter",
            "query": query,
            "count": len(processed_data),
            "data": processed_data
        })
        
    except Exception as e:
        app.logger.error(f"Twitter ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/reddit", methods=["POST"])
def ingest_reddit():
    """Ingest data from Reddit API"""
    try:
        data = request.json
        subreddit = data.get("subreddit", "")
        post_type = data.get("post_type", "hot")
        limit = data.get("limit", 100)
        
        if not subreddit:
            return jsonify({"error": "Subreddit is required"}), 400
        
        # Ingest data
        ingested_data = data_manager.ingest_from_reddit(subreddit, post_type, limit)
        
        if not ingested_data:
            return jsonify({"error": "No data ingested from Reddit"}), 404
        
        # Process with hate speech detection
        processed_data = data_manager.process_ingested_data(ingested_data, model)
        
        return jsonify({
            "status": "success",
            "source": "reddit",
            "subreddit": subreddit,
            "count": len(processed_data),
            "data": processed_data
        })
        
    except Exception as e:
        app.logger.error(f"Reddit ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/facebook", methods=["POST"])
def ingest_facebook():
    """Ingest data from Facebook Graph API"""
    try:
        data = request.json
        page_id = data.get("page_id", "")
        post_limit = data.get("post_limit", 50)
        
        if not page_id:
            return jsonify({"error": "Page ID is required"}), 400
        
        # Ingest data
        ingested_data = data_manager.ingest_from_facebook(page_id, post_limit)
        
        if not ingested_data:
            return jsonify({"error": "No data ingested from Facebook"}), 404
        
        # Process with hate speech detection
        processed_data = data_manager.process_ingested_data(ingested_data, model)
        
        return jsonify({
            "status": "success",
            "source": "facebook",
            "page_id": page_id,
            "count": len(processed_data),
            "data": processed_data
        })
        
    except Exception as e:
        app.logger.error(f"Facebook ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/web-scraping", methods=["POST"])
def ingest_web_scraping():
    """Ingest data from web scraping"""
    try:
        data = request.json
        urls = data.get("urls", [])
        selector = data.get("selector", None)
        
        if not urls:
            return jsonify({"error": "URLs are required"}), 400
        
        # Ingest data
        ingested_data = data_manager.ingest_from_web_scraping(urls, selector)
        
        if not ingested_data:
            return jsonify({"error": "No data ingested from web scraping"}), 404
        
        # Process with hate speech detection
        processed_data = data_manager.process_ingested_data(ingested_data, model)
        
        return jsonify({
            "status": "success",
            "source": "web_scraping",
            "urls_count": len(urls),
            "count": len(processed_data),
            "data": processed_data
        })
        
    except Exception as e:
        app.logger.error(f"Web scraping ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/csv", methods=["POST"])
def ingest_csv():
    """Ingest data from CSV file upload"""
    try:
        if "csv_file" not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400
        
        file = request.files["csv_file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        file.save(temp_path)
        
        # Get parameters
        text_column = request.form.get("text_column", "text")
        additional_columns = request.form.getlist("additional_columns")
        
        # Ingest data
        ingested_data = data_manager.ingest_from_csv(temp_path, text_column, additional_columns)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        if not ingested_data:
            return jsonify({"error": "No data ingested from CSV"}), 404
        
        # Process with hate speech detection
        processed_data = data_manager.process_ingested_data(ingested_data, model)
        
        return jsonify({
            "status": "success",
            "source": "csv",
            "filename": filename,
            "count": len(processed_data),
            "data": processed_data
        })
        
    except Exception as e:
        app.logger.error(f"CSV ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ingest/batch", methods=["POST"])
def ingest_batch():
    """Batch ingest from multiple sources"""
    try:
        data = request.json
        sources = data.get("sources", [])
        
        if not sources:
            return jsonify({"error": "No sources specified"}), 400
        
        all_processed_data = []
        results = {}
        
        for source_config in sources:
            source_type = source_config.get("type")
            
            try:
                if source_type == "twitter":
                    ingested = data_manager.ingest_from_twitter(
                        source_config.get("query", ""),
                        source_config.get("count", 100),
                        source_config.get("language", "en")
                    )
                elif source_type == "reddit":
                    ingested = data_manager.ingest_from_reddit(
                        source_config.get("subreddit", ""),
                        source_config.get("post_type", "hot"),
                        source_config.get("limit", 100)
                    )
                elif source_type == "facebook":
                    ingested = data_manager.ingest_from_facebook(
                        source_config.get("page_id", ""),
                        source_config.get("post_limit", 50)
                    )
                elif source_type == "web_scraping":
                    ingested = data_manager.ingest_from_web_scraping(
                        source_config.get("urls", []),
                        source_config.get("selector")
                    )
                else:
                    results[source_type] = {"status": "error", "error": "Unknown source type"}
                    continue
                
                # Process with hate speech detection
                processed = data_manager.process_ingested_data(ingested, model)
                all_processed_data.extend(processed)
                
                results[source_type] = {
                    "status": "success",
                    "ingested_count": len(ingested),
                    "processed_count": len(processed)
                }
                
            except Exception as e:
                results[source_type] = {"status": "error", "error": str(e)}
        
        return jsonify({
            "status": "success",
            "total_processed": len(all_processed_data),
            "results": results,
            "data": all_processed_data
        })
        
    except Exception as e:
        app.logger.error(f"Batch ingestion error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/export/data", methods=["POST"])
def export_data():
    """Export processed data to file"""
    try:
        data = request.json
        export_data = data.get("data", [])
        output_path = data.get("output_path", "exported_data")
        format_type = data.get("format", "csv")
        
        if not export_data:
            return jsonify({"error": "No data to export"}), 400
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_path}_{timestamp}.{format_type}"
        
        # Export data
        data_manager.export_processed_data(export_data, filename, format_type)
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "count": len(export_data),
            "format": format_type
        })
        
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Speech-to-Text only route (no hate speech detection)
@app.route("/transcribe", methods=["POST"])
def transcribe_audio_only():
    """Transcribe audio without hate speech detection"""
    
    # Check if file was uploaded
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files["audio"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Allowed types: wav, mp3, flac, m4a, ogg, webm"}), 400
    
    try:
        # Save uploaded audio file
        audio_path = speech_processor.save_uploaded_audio(file)
        
        if not audio_path:
            return jsonify({"error": "Failed to save audio file"}), 500
        
        # Transcribe audio
        method = request.form.get("method", "whisper")
        language = request.form.get("language")
        transcribed_text = speech_processor.transcribe_audio(audio_path, method, language=language)
        
        # Clean up temporary file
        speech_processor.cleanup_temp_file(audio_path)
        
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio"}), 500
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "method_used": method,
            "language_used": language
        })
        
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


# Start the Flask web application
if __name__ == "__main__":
    app.run(debug=True)
