#!/usr/bin/env python
import sys
import os
import warnings
import requests
import logging
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from restructurer_math.crew import Restructure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Django API configuration
DJANGO_API_BASE_URL = "https://transback.transpoze.ai"

# ---
# ### 🔍 Function to Retrieve Answer Sheet Data from OCR Endpoint
# ---
def get_answersheet_from_ocr(script_id: str):
    """Retrieve answer sheet data from Django API using OCR endpoint."""
    try:
        logger.info(f"Requesting OCR data for script_id: {script_id}")
        
        url = f"{DJANGO_API_BASE_URL}/ocr/?script_id={script_id}"
        logger.info(f"OCR API URL: {url}")         
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                # Combine all pages' OCR data into a single answer sheet
                combined_answersheet = ""
                combined_structured_data = {}
                
                # Sort by page number to maintain order
                sorted_data = sorted(data, key=lambda x: x.get('page_number', 0))
                
                for page_data in sorted_data:
                    page_number = page_data.get('page_number', 0)
                    ocr_json = page_data.get('ocr_json', {})
                    structured_json = page_data.get('structured_json', {})
                    context = page_data.get('context', '')
                    
                    # Extract text from OCR JSON (adjust based on your OCR format)
                    page_text = extract_text_from_ocr_json(ocr_json)
                    
                    # Add page separator and content
                    if combined_answersheet:
                        combined_answersheet += f"\n\n--- Page {page_number} ---\n"
                    else:
                        combined_answersheet += f"--- Page {page_number} ---\n"
                    
                    combined_answersheet += page_text
                    
                    # Add context if available
                    if context:
                        combined_answersheet += f"\n[Context: {context}]"
                    
                    # Combine structured data
                    combined_structured_data[f"page_{page_number}"] = structured_json
                
                if combined_answersheet:
                    logger.info(f"Successfully combined OCR data from {len(sorted_data)} pages for script_id: {script_id}")
                    return combined_answersheet, combined_structured_data, None
                else:
                    return None, None, f"No text content found in OCR data for script_id: {script_id}"
            else:
                return None, None, f"No OCR records found for script_id: {script_id}"
        else:
            logger.error(f"OCR API error: {response.status_code} - {response.text}")
            return None, None, f"OCR API error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR API request error: {str(e)}")
        return None, None, f"OCR API request error: {str(e)}"

def extract_text_from_ocr_json(ocr_json):
    """Extract readable text from OCR JSON data."""
    try:
        if not ocr_json:
            return ""
        
        # This function needs to be adapted based on your OCR JSON structure
        # Common OCR formats include AWS Textract, Google Vision, etc.
        
        if isinstance(ocr_json, dict):
            # Try common OCR JSON structures
            
            # AWS Textract format
            if 'Blocks' in ocr_json:
                return extract_from_textract_format(ocr_json)
            
            # Google Vision format
            elif 'textAnnotations' in ocr_json:
                return extract_from_google_vision_format(ocr_json)
            
            # Custom format - look for common text fields
            elif 'text' in ocr_json:
                return str(ocr_json['text'])
            
            # If structured as lines/paragraphs
            elif 'lines' in ocr_json:
                lines = ocr_json['lines']
                if isinstance(lines, list):
                    return '\n'.join([str(line.get('text', '')) for line in lines if line.get('text')])
            
            # Fallback: try to extract any text-like values
            else:
                return extract_text_recursively(ocr_json)
        
        elif isinstance(ocr_json, str):
            return ocr_json
        
        else:
            return str(ocr_json)
            
    except Exception as e:
        logger.warning(f"Error extracting text from OCR JSON: {str(e)}")
        return str(ocr_json) if ocr_json else ""

def extract_from_textract_format(ocr_json):
    """Extract text from AWS Textract format."""
    try:
        blocks = ocr_json.get('Blocks', [])
        text_blocks = []
        
        for block in blocks:
            if block.get('BlockType') == 'LINE' and 'Text' in block:
                text_blocks.append(block['Text'])
        
        return '\n'.join(text_blocks)
    except Exception as e:
        logger.warning(f"Error extracting from Textract format: {str(e)}")
        return ""

def extract_from_google_vision_format(ocr_json):
    """Extract text from Google Vision format."""
    try:
        annotations = ocr_json.get('textAnnotations', [])
        if annotations and len(annotations) > 0:
            # First annotation usually contains the full text
            return annotations[0].get('description', '')
        return ""
    except Exception as e:
        logger.warning(f"Error extracting from Google Vision format: {str(e)}")
        return ""

def extract_text_recursively(data, max_depth=3, current_depth=0):
    """Recursively extract text from nested JSON structures."""
    if current_depth >= max_depth:
        return ""
    
    text_parts = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                # Skip keys that are likely metadata
                if key.lower() not in ['id', 'type', 'confidence', 'bbox', 'coordinates']:
                    text_parts.append(value.strip())
            elif isinstance(value, (dict, list)):
                nested_text = extract_text_recursively(value, max_depth, current_depth + 1)
                if nested_text:
                    text_parts.append(nested_text)
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and len(item.strip()) > 0:
                text_parts.append(item.strip())
            elif isinstance(item, (dict, list)):
                nested_text = extract_text_recursively(item, max_depth, current_depth + 1)
                if nested_text:
                    text_parts.append(nested_text)
    
    return '\n'.join(text_parts)

# ---
# ### 🔍 Function to Retrieve VLMDesc from Compare-Text Endpoint
# ---
def get_vlmdesc_data(script_id: str):
    """Retrieve vlmdesc from Django API using compare-text endpoint."""
    try:
        logger.info(f"Requesting vlmdesc for script_id: {script_id}")
        
        url = f"{DJANGO_API_BASE_URL}/compare-text/?script_id={script_id}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                latest_record = data[0]
                vlmdesc = latest_record.get('vlmdesc')
                
                if vlmdesc:
                    logger.info(f"Found vlmdesc for script_id: {script_id}")
                    return vlmdesc, None
                else:
                    logger.warning(f"No vlmdesc found for script_id: {script_id}")
                    return {}, None
            else:
                logger.warning(f"No compare text records found for script_id: {script_id}")
                return {}, None
        else:
            logger.error(f"Compare-text API error for vlmdesc: {response.status_code} - {response.text}")
            return {}, f"Compare-text API error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Compare-text API request error for vlmdesc: {str(e)}")
        return {}, f"Compare-text API request error: {str(e)}"

# ---
# ### 🔍 Function to Retrieve MCQ from Compare-Text Endpoint
# ---
def get_mcq_data(script_id: str):
    """Retrieve mcq from Django API using compare-text endpoint."""
    try:
        logger.info(f"Requesting mcq for script_id: {script_id}")
        
        url = f"{DJANGO_API_BASE_URL}/compare-text/?script_id={script_id}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                latest_record = data[0]
                mcq = latest_record.get('mcq')
                
                if mcq:
                    logger.info(f"Found mcq for script_id: {script_id}")
                    return mcq, None
                else:
                    logger.warning(f"No mcq found for script_id: {script_id}")
                    return {}, None
            else:
                logger.warning(f"No compare text records found for script_id: {script_id}")
                return {}, None
        else:
            logger.error(f"Compare-text API error for mcq: {response.status_code} - {response.text}")
            return {}, f"Compare-text API error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Compare-text API request error for mcq: {str(e)}")
        return {}, f"Compare-text API request error: {str(e)}"

# ---
# ### 🔍 Function to Retrieve Rubrics from Key-OCR Endpoint
# ---
def get_rubrics_from_keyocr(subject_id: str):
    """Retrieve rubrics from Django API using key-ocr endpoint."""
    try:
        logger.info(f"Requesting rubrics for subject_id: {subject_id}")
        
        url = f"{DJANGO_API_BASE_URL}/key-ocr/?subject_id={subject_id}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, dict):
                rubrics = data.get('rubrics', '')
                if rubrics:
                    logger.info(f"Found rubrics for subject_id: {subject_id}")
                    return rubrics, None
                else:
                    return None, f"No rubrics found for subject_id: {subject_id}"
            else:
                return None, f"Unexpected response format for subject_id: {subject_id}"
        else:
            logger.error(f"Key-OCR API error: {response.status_code} - {response.text}")
            return None, f"Key-OCR API error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Key-OCR API request error: {str(e)}")
        return None, f"Key-OCR API request error: {str(e)}"

# ---
# ### 💾 Function to Save Result to Database
# ---
def save_result_to_database(script_id: str, restructured_data: dict):
    """Save the restructured result to the Django database."""
    try:
        logger.info(f"Saving result to database for script_id: {script_id}")
        
        url = f"{DJANGO_API_BASE_URL}/results/"
        
        payload = {
            "script_id": script_id,
            "restructuredtext": restructured_data,
            "scored": {},  # Initialize empty - will be filled by scoring service
            "graded": {},  # Initialize empty - will be filled by grading service
            "analytics": {}  # Initialize empty - will be filled by analytics service
        }
        
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result_data = response.json()
            logger.info(f"Successfully saved result to database: {result_data}")
            return True, result_data, None
        else:
            logger.error(f"Database save error: {response.status_code} - {response.text}")
            return False, None, f"Database save error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Database save request error: {str(e)}")
        return False, None, f"Database save request error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error saving to database: {str(e)}")
        return False, None, f"Unexpected error saving to database: {str(e)}"

def update_result_in_database(script_id: str, restructured_data: dict):
    """Update existing result in the Django database."""
    try:
        logger.info(f"Updating result in database for script_id: {script_id}")
        
        # First, get the result_id for this script
        get_url = f"{DJANGO_API_BASE_URL}/results/?script_id={script_id}"
        get_response = requests.get(get_url)
        
        if get_response.status_code == 200:
            results = get_response.json()
            if isinstance(results, list) and len(results) > 0:
                result_id = results[0]['result_id']
                
                # Update the result
                update_url = f"{DJANGO_API_BASE_URL}/results/"
                payload = {
                    "result_id": result_id,
                    "restructuredtext": restructured_data
                }
                
                response = requests.put(update_url, json=payload, headers={'Content-Type': 'application/json'})
                
                if response.status_code == 200:
                    result_data = response.json()
                    logger.info(f"Successfully updated result in database: {result_data}")
                    return True, result_data, None
                else:
                    logger.error(f"Database update error: {response.status_code} - {response.text}")
                    return False, None, f"Database update error: {response.status_code} - {response.text}"
            else:
                return False, None, f"No existing result found for script_id: {script_id}"
        else:
            return False, None, f"Error fetching existing result: {get_response.status_code} - {get_response.text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Database update request error: {str(e)}")
        return False, None, f"Database update request error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error updating database: {str(e)}")
        return False, None, f"Unexpected error updating database: {str(e)}"

# ---
# ### 🎯 Function to Format Result as Question-Answer Pairs
# ---
def format_as_qa_pairs(result_text):
    """Format the restructure result as question-answer pairs."""
    try:
        # Try to parse as JSON first
        if isinstance(result_text, str):
            try:
                result_json = json.loads(result_text)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text and create a simple structure
                result_json = {"content": result_text}
        else:
            result_json = result_text
        
        # Create formatted Q&A pairs
        qa_pairs = []
        
        # If the result contains structured data, extract Q&A pairs
        if isinstance(result_json, dict):
            for key, value in result_json.items():
                if isinstance(value, dict):
                    # If value is a dict, it might contain question-answer structure
                    for sub_key, sub_value in value.items():
                        qa_pairs.append({
                            "question": f"{key} - {sub_key}",
                            "answer": str(sub_value)
                        })
                elif isinstance(value, list):
                    # If value is a list, format each item
                    for i, item in enumerate(value):
                        qa_pairs.append({
                            "question": f"{key} - Item {i+1}",
                            "answer": str(item)
                        })
                else:
                    # Simple key-value pair
                    qa_pairs.append({
                        "question": str(key),
                        "answer": str(value)
                    })
        elif isinstance(result_json, list):
            # If result is a list, format each item
            for i, item in enumerate(result_json):
                qa_pairs.append({
                    "question": f"Item {i+1}",
                    "answer": str(item)
                })
        else:
            # If it's just a string or other type
            qa_pairs.append({
                "question": "Restructured Content",
                "answer": str(result_json)
            })
        
        return {
            "total_pairs": len(qa_pairs),
            "qa_pairs": qa_pairs,
            "formatted_display": format_qa_display(qa_pairs),
            "metadata": {
                "processed_at": str(json.dumps({"timestamp": "now"})),
                "processing_method": "restructure_crew",
                "data_sources": ["ocr", "vlmdesc", "mcq", "rubrics"]
            }
        }
    
    except Exception as e:
        logger.error(f"Error formatting Q&A pairs: {str(e)}")
        return {
            "total_pairs": 1,
            "qa_pairs": [{"question": "Restructured Content", "answer": str(result_text)}],
            "formatted_display": f"Q: Restructured Content\nA: {str(result_text)}",
            "metadata": {
                "processed_at": str(json.dumps({"timestamp": "now"})),
                "processing_method": "restructure_crew",
                "error": str(e)
            }
        }

def format_qa_display(qa_pairs):
    """Create a formatted display string for Q&A pairs."""
    display_lines = []
    
    for i, pair in enumerate(qa_pairs, 1):
        display_lines.append(f"\n{'='*60}")
        display_lines.append(f"Question {i}:")
        display_lines.append(f"{'='*60}")
        display_lines.append(pair["question"])
        display_lines.append(f"\n{'-'*60}")
        display_lines.append("Answer:")
        display_lines.append(f"{'-'*60}")
        display_lines.append(pair["answer"])
    
    display_lines.append(f"\n{'='*60}")
    
    return "\n".join(display_lines)

# ---
# ### 🧠 Updated Core Restructure Logic
# ---
def run_restructure(subject_id: str, script_id: str):
    """Run restructure pipeline using subject_id and script_id."""
    try:
        # Get answer sheet data from OCR endpoint
        answersheet_text, structured_data, ocr_error = get_answersheet_from_ocr(script_id)
        if ocr_error:
            return False, ocr_error, None

        if not answersheet_text:
            return False, f"No answer sheet data found for script_id: {script_id}", None
        
        # Get VLM description data
        vlmdesc_data, vlmdesc_error = get_vlmdesc_data(script_id)
        if vlmdesc_error:
            logger.warning(f"VLMDesc retrieval error (will use fallback): {vlmdesc_error}")
            vlmdesc_data = {}
        
        # Get MCQ data
        mcq_data, mcq_error = get_mcq_data(script_id)
        if mcq_error:
            logger.warning(f"MCQ retrieval error (will use fallback): {mcq_error}")
            mcq_data = {}
        
        # Get rubrics
        rubrics, rubrics_error = get_rubrics_from_keyocr(subject_id)
        if rubrics_error:
            return False, rubrics_error, None

        if not rubrics:
            return False, f"No rubrics found for subject_id: {subject_id}", None

        # Prepare inputs for the agent
        inputs = {
            "answersheet": answersheet_text,
            "context": rubrics,
            "vlmdesc": vlmdesc_data,
            "mcq": mcq_data
        }

        # Log inputs
        logger.info(f"Answer sheet text length: {len(answersheet_text)}")
        logger.info(f"VLMDesc: {str(vlmdesc_data)[:100]}...")
        logger.info(f"MCQ: {str(mcq_data)[:100]}...")

        # Print extracted data
        print("\n" + "="*80)
        print("📄 EXTRACTED ANSWER SHEET FROM OCR:")
        print("="*80)
        print(answersheet_text)
        print("\n" + "="*80)
        print("📊 STRUCTURED OCR DATA:")
        print("="*80)
        print(json.dumps(structured_data, indent=2) if structured_data else "No structured data")
        print("\n" + "="*80)
        print("📝 RUBRICS:")
        print("="*80)
        print(rubrics)
        print("\n" + "="*80)
        print("🔍 VLMDESC:")
        print("="*80)
        print(vlmdesc_data)
        print("\n" + "="*80)
        print("📊 MCQ:")
        print("="*80)
        print(mcq_data)
        print("="*80 + "\n")

        # Run the agent
        result = Restructure().crew().kickoff(inputs=inputs)
        print(f"\nToken Usage:\n{result.token_usage}\n")
        
        # Convert result to string
        result_text = str(result)
        
        # Format result as Q&A pairs
        formatted_result = format_as_qa_pairs(result_text)
        
        # Print formatted Q&A pairs
        print("\n" + "="*80)
        print("🎯 FORMATTED QUESTION-ANSWER PAIRS:")
        print("="*80)
        print(formatted_result["formatted_display"])
        print("="*80 + "\n")

        # Save to database
        save_success, save_result, save_error = save_result_to_database(script_id, formatted_result)
        
        if not save_success:
            # If save failed, try to update existing record
            logger.warning(f"Save failed, trying to update existing record: {save_error}")
            update_success, update_result, update_error = update_result_in_database(script_id, formatted_result)
            
            if update_success:
                logger.info(f"Successfully updated existing result for script_id: {script_id}")
                database_operation = "updated"
                database_result = update_result
            else:
                logger.error(f"Both save and update failed for script_id: {script_id}")
                database_operation = "failed"
                database_result = {"error": update_error}
        else:
            logger.info(f"Successfully saved new result for script_id: {script_id}")
            database_operation = "created"
            database_result = save_result

        logger.info(f"Restructure completed successfully for script_id: {script_id}")
        
        # Include database operation info in response
        formatted_result["database_operation"] = database_operation
        formatted_result["database_result"] = database_result
        
        return True, f"Success: Restructure completed for script_id {script_id} and {database_operation} in database", formatted_result

    except Exception as e:
        logger.error(f"Restructure failed: {str(e)}")
        return False, f"Error: {str(e)}", None

# ---
# ### 🚀 Flask Application
# ---
def run():
    """Start the Flask application for restructure."""
    app = Flask(__name__)
    app.secret_key = 'restructure_secret_key'
    
    # CORS configuration
    cors_config = {
        "origins": [
            "http://localhost:3000",
            "https://transgrade.transpoze.ai",
            "http://localhost:3001",
            "https://transpoze.ai",
            "https://*.transpoze.ai",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "https://localhost:3000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ['*'],
        "supports_credentials": True
    }
    
    CORS(app, **cors_config)
    
    @app.route('/mathres/')
    def index():
        """Root endpoint with API information."""
        return jsonify({
            "message": "Restructure API is running",
            "endpoints": {
                "restructure": "/restructure/restructure/<subject_id>/<script_id>",
                "health_check": "/restructure/health"
            }
        })

    @app.route('/mathres/restructure/<subject_id>/<script_id>', methods=['GET'])
    def restructure_route(subject_id, script_id):
        """Endpoint to run restructure for a given subject_id and script_id."""
        if not subject_id or not script_id:
            return jsonify({
                "status": "error", 
                "message": "Subject ID and Script ID are required"
            }), 400
        
        logger.info(f"Processing restructure for subject_id: {subject_id}, script_id: {script_id}")
        success, message, formatted_result = run_restructure(subject_id, script_id)
        
        response_data = {
            "status": "success" if success else "error",
            "subject_id": str(subject_id),
            "script_id": str(script_id),
            "message": message
        }
        
        if success and formatted_result:
            response_data["result"] = formatted_result
        
        return jsonify(response_data), 200 if success else 500

    @app.route('/mathres/health')
    def health_check():
        """Health check endpoint to verify Django API connectivity."""
        try:
            response = requests.get(f"{DJANGO_API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                return jsonify({
                    "status": "healthy", 
                    "django_api": "connected",
                    "django_url": DJANGO_API_BASE_URL
                })
            else:
                return jsonify({
                    "status": "unhealthy", 
                    "django_api": "error", 
                    "details": f"Status: {response.status_code}"
                })
        except Exception as e:
            return jsonify({
                "status": "unhealthy", 
                "django_api": "disconnected", 
                "error": str(e)
            })

   
    
    # Handle OPTIONS requests for CORS preflight
    @app.before_request
    def handle_preflight():        
        if request.method == "OPTIONS":
            response = jsonify({})
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response

    # Run Flask app
    port = int(os.environ.get('PORT', 8888))
    app.run(host='0.0.0.0', port=port, debug=False)

# ---
# ### 🧭 Main Entry Point
# ---

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "run":
        run()
    else:
        print("Usage: python main.py run")