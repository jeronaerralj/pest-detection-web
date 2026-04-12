from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify, Response, make_response
import sqlite3, os, cv2, threading, atexit, re, datetime, time, json, shutil, sys
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import numpy as np 
import base64 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics.models.yolo import YOLO 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import hashlib
import uuid
from google import genai
from groq import Groq 
from openai import OpenAI
import ollama
import PIL.Image 
from dotenv import load_dotenv
from functools import wraps
from datetime import timedelta
from urllib.parse import urlparse
import csv
import io
import serial

load_dotenv()

# ================== EMAIL CONFIG ==================

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
GMAIL_NOTIFICATION_EMAIL = os.getenv("GMAIL_NOTIFICATION_EMAIL")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_fallback_key')
app.permanent_session_lifetime = timedelta(seconds=int(os.getenv('SESSION_TIMEOUT_SEC', '900')))

PEST_ALIASES = {
    "Mealybug": "Pineapple Mealybug",
    "mealybug": "Pineapple Mealybug",
    "mealy bug": "Pineapple Mealybug",
    "Mealybug Cluster": "Pineapple Mealybug", # Added
    "Giant African Snail": "Giant African Land Snail",
    "African Snail": "Giant African Land Snail",
    "Oryctes Rhinoceros Beetle": "Rhinoceros Beetle",
    "Coconut Rhinoceros Beetle": "Rhinoceros Beetle",
    "Fruit Fly": "Oriental Fruit Fly",
    "Cut worm": "Cutworm",
    "Cutworm Larva": "Cutworm",
    "Cutworm Moth": "Cutworm",
    "Coconut Slug Caterpillar": "Slug Caterpillar",
    "Asian Weaver Ant": "Weaver Ant",
    "Weaver Ant Cluster": "Weaver Ant",
    "Gray Borer Generic": "Gray Borer"
}

# ================== DIRECTORY CONFIGURATION ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database paths
DB_DIR = os.path.join(BASE_DIR, 'database')
DATABASE = os.path.join(DB_DIR, 'pests_add.db')

# Static image folders (for the web interface)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
HISTORY_FOLDER = os.path.join(BASE_DIR, 'static', 'history')

# Active Learning dataset folders (for YOLO retraining)
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_IMG_DIR = os.path.join(DATASET_DIR, 'images')
DATASET_LBL_DIR = os.path.join(DATASET_DIR, 'labels')
ACTIVE_LEARNING_CLASSES_FILE = os.path.join(DATASET_DIR, 'classes.json')

# Tell Flask where uploads go
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Automatically create these folders if they don't exist yet
for folder in [DB_DIR, UPLOAD_FOLDER, HISTORY_FOLDER, DATASET_IMG_DIR, DATASET_LBL_DIR]:
    os.makedirs(folder, exist_ok=True)
# --- SERIAL CONFIGURATION ---
SERIAL_PORT = 'COM4' # Change to your Arduino Port (e.g., /dev/ttyUSB0 on Linux)
BAUD_RATE = 9600
arduino = None
arduino_data = {"uv": 0, "light": "OFF", "mode": "AUTO"}

# ============= LIGHT LOGIC USING ARDUINO ==================

def init_serial():
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Arduino Connected on {SERIAL_PORT}")
        # Start background thread to read data
        thread = threading.Thread(target=read_from_arduino)
        thread.daemon = True
        thread.start()
    except Exception as e:
        print(f"⚠️ Arduino Connection Failed: {e}")

def read_from_arduino():
    global arduino_data
    while True:
        if arduino and arduino.in_waiting > 0:
            try:
                line = arduino.readline().decode('utf-8').strip()
                if line.startswith('{'): # Simple check for JSON
                    arduino_data = json.loads(line)
            except Exception:
                pass
        time.sleep(0.1)

# Call this before app.run()
init_serial()

# --- NEW ROUTES FOR LIGHT CONTROL ---

@app.route('/api/light/status')
def get_light_status():
    return jsonify(arduino_data)

@app.route('/api/light/control', methods=['POST'])
def control_light():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    action = request.json.get('action')  # type: ignore
    if not arduino:
        return jsonify({'success': False, 'error': 'Arduino not connected'})
    
    command = ""
    if action == 'on': command = "LIGHT_ON\n"
    elif action == 'off': command = "LIGHT_OFF\n"
    elif action == 'auto': command = "AUTO_MODE\n"
    
    if command:
        arduino.write(command.encode())
        return jsonify({'success': True, 'status': action})
    
    return jsonify({'success': False, 'error': 'Invalid command'})

# ================== AI CONFIGURATION ==================
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY') 
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN') 
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') # Added OpenRouter Key

if not GENAI_API_KEY:
    print("⚠️ WARNING: GENAI_API_KEY not found in .env file")
else:
    try:
        gemini_client = genai.Client(api_key=GENAI_API_KEY)
        print("✅ Gemini AI Configured Successfully")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")

if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found (Groq fallback disabled)")

if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found (OpenRouter fallback disabled)")

if not GITHUB_TOKEN:
    print("⚠️ WARNING: GITHUB_TOKEN not found (GitHub Vision fallback disabled)")

# ================== GLOBAL STATE & LOCKS (UPDATED FOR MULTI-CAM) ==================
frame_lock = threading.Lock()
db_lock = threading.Lock()

is_detection_running = False      

# Individual tracking for each camera
cam_states = {
    "CAM 1": {"detected_pest": "", "timeout": 0, "confidence": 0.0, "ai_override": None, "ai_processing": False, "ai_cache": None, "last_request": 0, "last_logged": None, "last_log_time": 0, "last_email_time": 0},
    "CAM 2": {"detected_pest": "", "timeout": 0, "confidence": 0.0, "ai_override": None, "ai_processing": False, "ai_cache": None, "last_request": 0, "last_logged": None, "last_log_time": 0, "last_email_time": 0},
    "CAM 3": {"detected_pest": "", "timeout": 0, "confidence": 0.0, "ai_override": None, "ai_processing": False, "ai_cache": None, "last_request": 0, "last_logged": None, "last_log_time": 0}
}
# NEW: Track how many times a pest is confidently seen before approving it
temporal_counts = {
    "CAM 1": {}, "CAM 2": {}, "CAM 3": {}
}

# ================== ACTIVE LEARNING HELPERS ==================
BASE_CLASS_COUNT = 12

def get_or_create_class_id(pest_name):
    """Assigns a permanent integer ID to a new pest for YOLO training."""
    classes = {}
    if os.path.exists(ACTIVE_LEARNING_CLASSES_FILE):
        with open(ACTIVE_LEARNING_CLASSES_FILE, 'r') as f:
            classes = json.load(f)
    
    if pest_name not in classes:
        new_id = BASE_CLASS_COUNT + len(classes)
        classes[pest_name] = new_id
        with open(ACTIVE_LEARNING_CLASSES_FILE, 'w') as f:
            json.dump(classes, f)
    
    return classes[pest_name]

def check_dataset_threshold(pest_name, threshold=50):
    """Checks if we have gathered enough images of this pest to trigger a retrain."""
    class_id = get_or_create_class_id(pest_name)
    count = 0
    for txt_file in os.listdir(DATASET_LBL_DIR):
            if txt_file.endswith(".txt"):
                filepath = os.path.join(DATASET_LBL_DIR, txt_file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read().strip()
                        if content:
                            # Split by whitespace and check the first element (the ID)
                            first_val = content.split()[0]
                            if first_val == str(class_id):
                                count += 1
                except (IndexError, IOError):
                    continue
    
    if count >= threshold:
        print(f"🌟 ACTIVE LEARNING ALERT: Accumulated {count} images for {pest_name}! Ready for retraining.")

# ================== GRAPH ====================

@app.route('/api/analytics/pest-options')
def get_pest_options():
    try:
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT yolo_name FROM history ORDER BY yolo_name ASC")
        pests = [row[0] for row in cur.fetchall()]
        conn.close()
        return jsonify({'success': True, 'pests': pests})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/pest-timeline')
def get_pest_timeline():
    pest_name = request.args.get('pest')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    # Default to past 30 days if no dates are manually selected
    if not start_date or not end_date:
        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=30)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')

    query = "SELECT strftime('%Y-%m-%d', timestamp) as date, COUNT(*) FROM history WHERE yolo_name = ?"
    params = [pest_name]
    
    query += " AND timestamp BETWEEN ? AND ?"
    params.extend([f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
    
    query += " GROUP BY date ORDER BY date ASC"
    
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    
    return jsonify({
        'success': True, 
        'labels': [row[0] for row in rows],  
        'values': [row[1] for row in rows]   
    })

@app.route('/api/analytics/pest-frequency')
def get_pest_frequency():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = None
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Default to past 30 days if no dates are manually selected
        if not start_date or not end_date:
            end_dt = datetime.datetime.now()
            start_dt = end_dt - datetime.timedelta(days=30)
            start_date = start_dt.strftime('%Y-%m-%d')
            end_date = end_dt.strftime('%Y-%m-%d')

        query = "SELECT yolo_name, COUNT(*) as count FROM history WHERE timestamp BETWEEN ? AND ? GROUP BY yolo_name ORDER BY count DESC"
        params = [f"{start_date} 00:00:00", f"{end_date} 23:59:59"]
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        labels = [row['yolo_name'] for row in results]
        values = [row['count'] for row in results]
        
        return jsonify({'success': True, 'labels': labels, 'values': values})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn: conn.close()

@app.route('/api/analytics/export-csv')
def export_pest_csv():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    query = "SELECT timestamp, yolo_name, detection_type, user_session FROM history"
    params = []
    if start_date and end_date:
        query += " WHERE timestamp BETWEEN ? AND ?"
        params.extend([f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
    
    cur.execute(query, params)
    rows = cur.fetchall()
    
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Timestamp', 'Pest Name', 'Detection Method', 'Logged By'])
    cw.writerows(rows)
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename=pest_report_{datetime.date.today()}.csv"
    output.headers["Content-type"] = "text/csv"
    return output

# ================== HELPERS ==================

def clean_json_text(text):
    try:
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```', '', text)
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            text = text[start:end]
        return text.strip()
    except Exception as e:
        print(f"Error cleaning JSON: {e}")
        return "{}"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()
# ================== EMAIL NOTIFICATION ==================

# ================== EMAIL NOTIFICATION ==================

# Global variable to store email safely across threads
ACTIVE_USER_EMAIL = None

def _send_email_thread(pest_name, camera_id, target_email):
    try:
        if not EMAIL_SENDER or not EMAIL_PASSWORD or not target_email:
            return

        subject = f"🚨 Pest Detected: {pest_name}"
        body = f"Pest Detection Alert\n\nPest: {pest_name}\nCamera: {camera_id}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nThis alert was generated automatically by the AI Pest Detection System."

        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = target_email
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, target_email, msg.as_string())
        server.quit()

        print(f"📧 Email sent to {target_email}: {pest_name} detected on {camera_id}")

    except Exception as e:
        print(f"❌ Email error: {e}")

def send_pest_email(pest_name, camera_id):
    global ACTIVE_USER_EMAIL
    # Prioritize the email the user typed in, fallback to .env default
    email_to_use = ACTIVE_USER_EMAIL or GMAIL_NOTIFICATION_EMAIL
    
    if email_to_use:
        # Spawn a background thread so the camera feed NEVER freezes waiting for Gmail
        threading.Thread(target=_send_email_thread, args=(pest_name, camera_id, email_to_use)).start()

# --- STRICT URL ACCESS RESTRICTION ---
def restrict_url_access(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        referrer = request.headers.get('Referer')
        if not referrer:
            return redirect(url_for('home'))
        referrer_host = urlparse(referrer).netloc
        request_host = request.host
        
        if referrer_host != request_host:
            session.clear()
            return redirect(url_for('home'))
        referrer_path = urlparse(referrer).path
        allowed_referrers = [
            '/admin_dashboard', '/add_pest', '/delete_pest', '/upload_pest_image',
            '/update_pests', '/register', '/pest_list', '/login', '/user',
            '/index', '/upload', '/library'
        ]
        
        is_allowed = False
        for allowed in allowed_referrers:
            if referrer_path.startswith(allowed) or referrer_path == allowed:
                is_allowed = True
                break
        
        if not is_allowed:
            return redirect(url_for('home'))
        
        return f(*args, **kwargs)
    return decorated

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        is_api = request.path.startswith('/api') or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', '')
        
        if 'admin' not in session:
            if is_api:
                return jsonify({'success': False, 'error': 'Authentication required'}), 403
            flash("Please log in to access that page.", "warning")
            return redirect(url_for('login'))
        
        if 'session_token' not in session:
            session.clear()
            if is_api:
                return jsonify({'success': False, 'error': 'Session validation failed'}), 403
            flash("Your session is invalid. Please log in again.", "warning")
            return redirect(url_for('login'))
        
        last = session.get('last_activity')
        timeout = int(os.getenv('SESSION_TIMEOUT_SEC', '900'))
        now = time.time()
        if last and (now - last) > timeout:
            session.clear()
            if is_api:
                return jsonify({'success': False, 'error': 'Session expired'}), 403
            flash("Your session has expired. Please log in again.", "warning")
            return redirect(url_for('login'))
        
        session['last_activity'] = now
        return f(*args, **kwargs)
    return decorated

# ================== AI LOGIC (THREADED) ==================

def fetch_pest_info_from_ai(pest_name, image_path=None):
    json_structure = {
        "type": "Non-Native Species",
        "common_name": "Standard Name",
        "scientific_name": "Latin Name",
        "classification": "Insect/Animal", # Removed Fungi/Etc to keep it strict
        "family": "Family Name",
        "order_name": "Order Name",
        "cultural_methods": "Preventative farming practices 1-2 sentences",
        "biological_control": "Natural predators or biological agents 1-2 sentence",
        "sanitation": "Cleaning and removal advice 1-2 sentence",
        "mechanical_control": "Physical traps or barriers 1-2 sentence",
        "chemical_control": "Pesticides or chemical deterrents 1-2 sentence",
        "confidence_score": 95
    }

    # --- THE STRICT VISION PROMPT ---
    vision_prompt = f"""
    You are an expert AI Pest Specialist for Pineapple Farming. 
    Analyze the image and identify the specific INSECT, BUG, or ANIMAL PEST.
    
    CRITICAL RULES:
    1. DO NOT identify plants, leaves, crops, or inanimate objects.
    2. DO NOT identify plant diseases, viruses, fungi, rot, or wilt.
    3. If the subject is a plant, leaf, crop, or plant disease, you MUST return "common_name": "N/A".
    4. Return ONLY valid JSON in this exact format: {json.dumps(json_structure)}
    "confidence_score" MUST be an integer from 0 to 100 representing your certainty. If the image is blurry, dark, or ambiguous, return a score below 50.
    """

    # --- SCENARIO A: VISION ANALYSIS ---
    if (pest_name.lower() in ["unknown", "negative"]) and image_path:
        print(f"🚀 AI Vision: Starting multi-model analysis for {image_path}...")

        # 1. PREPARE BASE64 (Do this once for all fallbacks)
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Critical Error: Could not encode image: {e}")
            return None

        # 2. Try Gemini Vision (Primary)
        if GENAI_API_KEY:
            # Using current supported models
            gemini_candidates = ['gemini-2.0-flash', 'gemini-1.5-flash']
            try:
                img = PIL.Image.open(image_path)
                for model_name in gemini_candidates:
                    try:
                        print(f"   ...Trying Gemini: {model_name}")
                        response = gemini_client.models.generate_content(
                            model=model_name,
                            contents=[vision_prompt, img]
                        )
                        return json.loads(clean_json_text(response.text))
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            print(f"   ⚠️ Quota Hit on {model_name}.")
                            continue 
                        print(f"   ❌ Error with {model_name}: {e}")
            except Exception as e:
                print(f"❌ Gemini Vision Critical Fail: {e}")

        # 3. Try OpenRouter Free Vision (Fallback 1)
        if OPENROUTER_API_KEY:
            print("   👉 Switching to OpenRouter (Free Vision)...")
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    temperature=0,
                )
                content = response.choices[0].message.content
                if content: return json.loads(clean_json_text(content))
            except Exception as e: print(f"OpenRouter failed: {e}")

        # 4. Try Groq Vision (Fallback 2)
        if GROQ_API_KEY:
            print("   👉 Switching to Groq (Llama 3.2 Vision)...")
            try:
                client = Groq(api_key=GROQ_API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    model="llama-4-scout-17b-16e-instruct",
                    temperature=0,
                    response_format={"type": "json_object"} 
                )
                content = chat_completion.choices[0].message.content
                if content: return json.loads(clean_json_text(content))
            except Exception as e: print(f"Groq Vision failed: {e}")

        # 5. Try GitHub Models (Fallback 3)
        if GITHUB_TOKEN:
            print("   👉 Switching to GitHub (Llama 3.2 90B)...")
            try:
                client = OpenAI(base_url="https://models.inference.ai.azure.com", api_key=GITHUB_TOKEN)
                response = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }],
                    model="Llama-3.2-90B-Vision-Instruct",
                    temperature=0,
                )
                content = response.choices[0].message.content
                if content: return json.loads(clean_json_text(content))
            except Exception as e: print(f"GitHub Models failed: {e}")

        # 6. Try Ollama (Local Backup)
        print("   👉 Final Fallback: Local Ollama...")
        try:
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{'role': 'user', 'content': vision_prompt, 'images': [image_path]}]
            )
            return json.loads(clean_json_text(response['message']['content']))
        except Exception as e: print(f"Ollama failed: {e}")

        print("❌ All AI Vision models failed.")
        return None

    # --- SCENARIO B: TEXT LOOKUP (If pest_name is already known) ---
    system_prompt = f"""
    You are an expert Pineapple agronomy AI. Provide management details for the pest '{pest_name}'.
    CRITICAL: If '{pest_name}' is a plant, crop, leaf, or plant disease, return "common_name": "N/A".
    Output ONLY valid JSON: {json.dumps(json_structure)}
    """

    if GENAI_API_KEY:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=system_prompt
            )
            return json.loads(clean_json_text(response.text))
        except: pass         

    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    # Replaced unreleased Llama 4 with Groq's active Llama 3.2 Vision model
                    model="llama-4-scout-17b-16e-instruct",
                    temperature=0,
                    # Note: response_format JSON mode is sometimes finicky on Groq vision previews, 
                    # but leaving it here is fine since your prompt strictly enforces JSON anyway.
                    response_format={"type": "json_object"} 
                )
            content = chat_completion.choices[0].message.content
            if content: return json.loads(clean_json_text(content))
        except Exception: pass
        
    return None

def start_ai_analysis_thread(frame_image, cam_id, x1, y1, x2, y2, frame_w, frame_h):
    try:
        temp_filename = f"unknown_{cam_id.replace(' ', '')}_{int(time.time())}.jpg"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        cv2.imwrite(temp_path, frame_image)
        process_unknown_pest_background(temp_path, cam_id, x1, y1, x2, y2, frame_w, frame_h)
        
    except Exception as e:
        print(f"❌ Thread Start Error: {e}")
        global cam_states
        cam_states[cam_id]["ai_processing"] = False

def process_unknown_pest_background(image_path, cam_id, x1, y1, x2, y2, frame_w, frame_h):
    global cam_states 
    
    print(f"🚀 Background Thread Started: Analyzing Unknown Pest on {cam_id}...")
    try:
        ai_data = fetch_pest_info_from_ai("Unknown", image_path=image_path)

# --- NEW: COMBINED SAFETY CHECKS ---
        if ai_data and ai_data.get('common_name') not in ["N/A", "Standard Name", None]:
            identified_name = ai_data.get('common_name').strip()
            
            # Extract the confidence score (default to 0 if the AI messes up)
            confidence = int(ai_data.get('confidence_score', 0))
            
            # CHECK 1: Is the AI confident enough?
            if confidence >= 85: 
                
                # Initialize or increment the temporal counter for this specific camera & pest
                counts = temporal_counts[cam_id]
                counts[identified_name] = counts.get(identified_name, 0) + 1
                current_count = counts[identified_name]
                
                print(f"🔍 AI saw {identified_name} on {cam_id} (Confidence: {confidence}%). Verification: {current_count}/3")
                
                # CHECK 2: Has it been seen 3 times?
                if current_count >= 3:
                    print(f"✅ FULLY VERIFIED: {identified_name} passed all temporal and confidence checks!")
                    
                    filename = os.path.basename(image_path)
                    log_detection_event(identified_name, f"uploads/{filename}", "Live AI Detection", confidence, camera_id=cam_id)
                    
                    cam_states[cam_id]["ai_cache"] = ai_data
                    cam_states[cam_id]["ai_override"] = identified_name
                    cam_states[cam_id]["detected_pest"] = identified_name
                    cam_states[cam_id]["last_logged"] = identified_name
                    cam_states[cam_id]["last_log_time"] = time.time()

                    # --- DB SAVE ---
                    try:
                        with db_lock:
                            conn = sqlite3.connect(DATABASE, timeout=30) 
                            c = conn.cursor()
                            db_image_path = f"uploads/{filename}"
                            c.execute("SELECT id FROM pests WHERE common_name = ?", (identified_name,))
                            if not c.fetchone():
                                c.execute('''INSERT INTO pests (type, 
                                    common_name, scientific_name, order_name, family, classification,
                                    cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image, yolo_name
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                                    "Non-Native Species", identified_name, 
                                    ai_data.get('scientific_name', 'N/A'), ai_data.get('order_name', 'N/A'), 
                                    ai_data.get('family', 'N/A'), ai_data.get('classification', 'Non-Native/Intruder'), 
                                    ai_data.get('cultural_methods', '—'), ai_data.get('biological_control', '—'), 
                                    ai_data.get('sanitation', '—'), ai_data.get('mechanical_control', '—'), 
                                    ai_data.get('chemical_control', '—'), db_image_path, identified_name 
                                ))
                                conn.commit()
                                print(f"💾 Saved '{identified_name}' to Database for future lookup.")
                            conn.close()
                    except Exception as db_e:
                        print(f"⚠️ Database Save Failed: {db_e}")

                    # --- ACTIVE LEARNING PIPELINE ---
                    try:
                        class_id = get_or_create_class_id(identified_name)
                        x_center = ((x1 + x2) / 2.0) / frame_w
                        y_center = ((y1 + y2) / 2.0) / frame_h
                        width_norm = (x2 - x1) / frame_w
                        height_norm = (y2 - y1) / frame_h
                        base_name = os.path.splitext(filename)[0]
                        
                        shutil.copy(image_path, os.path.join(DATASET_IMG_DIR, f"{base_name}.jpg"))
                        label_path = os.path.join(DATASET_LBL_DIR, f"{base_name}.txt")
                        with open(label_path, "w") as f:
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                            f.flush() # Forces the buffer to write to the file
                            os.fsync(f.fileno())
                        check_dataset_threshold(identified_name)
                    except Exception as al_err:
                        print(f"⚠️ Active Learning Accumulation Failed: {al_err}")

                    # Reset the counter so it doesn't instantly trigger again
                    temporal_counts[cam_id][identified_name] = 0
                
                else:
                    pass # High confidence, but waiting for 3rd strike
                    
            else:
                print(f"⚠️ AI Low Confidence ({confidence}%) for {identified_name}. Discarding to prevent false positive.")
    except Exception as e:
        print(f"❌ Critical Background Error on {cam_id}: {e}")
    finally:
        cam_states[cam_id]["ai_processing"] = False
# ================== DB INIT & PATCHING ==================
def init_databases():
    admin_db_path = os.path.join(DB_DIR, 'admin_db.db')
    conn = sqlite3.connect(admin_db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin'
        )
    """)
    conn.commit()
    conn.close()

    pest_db_path = os.path.join(DB_DIR, 'pests_add.db')
    conn = sqlite3.connect(pest_db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            common_name TEXT,
            scientific_name TEXT,
            order_name TEXT,
            family TEXT,
            classification TEXT,
            cultural_methods TEXT,
            biological_control TEXT,
            sanitation TEXT,
            mechanical_control TEXT,
            chemical_control TEXT,
            image TEXT,
            yolo_name TEXT 
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            yolo_name TEXT NOT NULL, 
            image_path TEXT,        
            user_session TEXT,      
            detection_type TEXT,
            confidence TEXT      
        )
    """)
    conn.commit()
    conn.close()
init_databases()

def patch_database_schema():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(pests)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'yolo_name' not in columns:
            cursor.execute("ALTER TABLE pests ADD COLUMN yolo_name TEXT")
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error patching database: {e}")

patch_database_schema()

# --- MODELS ---
try:
    model = YOLO('native.pt')
    print("✅ Custom Pest Model Loaded")
except:
    print("⚠️ WARNING: native.pt not found. Detection will fail.")
    model = None

# ================== LOGGING & CAMERA ==================
def log_detection_event(pest_name, image_path, detection_type, confidence=0.0, camera_id="CAM 1",):
    if confidence <= 1.0 and confidence > 0:
        confidence = confidence * 100
    with db_lock:
        try:
            conn = sqlite3.connect(DATABASE, timeout=10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                user = session.get('admin', 'SYSTEM') if 'session' in globals() else 'SYSTEM'
            except:
                user = 'SYSTEM'
            image_path = image_path.replace('\\', '/')
            conn.execute("""
                INSERT INTO history (timestamp, yolo_name, image_path, user_session, detection_type, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (current_time, f"[{camera_id}] {pest_name}", image_path, user, detection_type, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging history: {e}")
        # Send email notification
        threading.Thread(target=send_pest_email, args=(pest_name, camera_id)).start()

cams: dict[int, cv2.VideoCapture | None] = {0: None, 1: None, 2: None}

def get_camera(index):
    global cams
    if cams[index] is None or (cams[index] is not None and not cams[index].isOpened()):  # type: ignore
        cams[index] = cv2.VideoCapture(index) 
    return cams[index]

def get_blank_frame(text="CAMERA NOT FOUND"):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return blank

def is_detection_logical(label, box_w, box_h, frame_w, frame_h):
    area = box_w * box_h
    screen_area = frame_w * frame_h
    coverage = (area / screen_area) * 100
    
    short_side = min(box_w, box_h)
    long_side = max(box_w, box_h)
    ratio = long_side / short_side if short_side > 0 else 0

    if label == "Rhinoceros Beetle":
        if coverage < 1.5: return False 
        return True

    if label == "Cutworm Larva":
        if ratio < 1.3: return False 
        return True

    if label in ["Flower Thrips", "Mealybug"]:
        if coverage > 5.0: return False 
        return True

    if "Cluster" in label:
        if coverage > 40.0: return False 
        return True

    if label in ["Weaver Ant", "Oriental Fruit Fly"]:
        if coverage > 10.0: return False 
        return True

    if label in ["Gray Borer", "Cutworm Moth", "Gray Borer Generic"]:
        if coverage > 20.0: return False 
        return True

    return True 

def process_camera_frame(frame, cam_id):
    global cam_states
    state = cam_states[cam_id]

    annotated_frame = frame.copy()
    
    # --- Extracted frame width and height ---
    frame_h, frame_w = frame.shape[:2]
    
    pest_found_in_this_frame = False
    best_conf = 0.0
    best_pest = None

    moving_boxes = []  # Placeholder for motion detection boxes

    if model:
        # Run YOLO inference
        results = model(frame, stream=True, conf=0.25, verbose=False, agnostic_nms=True)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                raw_label = r.names[cls_id]
                
                # Apply your mapping (e.g., merging clusters or life stages)
                label = PEST_ALIASES.get(raw_label, raw_label)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if not is_detection_logical(raw_label, (x2-x1), (y2-y1), 640, 480):
                    continue

                # Scenario A: High Confidence Known Pest
                if conf > 0.70:
                    pest_found_in_this_frame = True
                    best_conf, best_pest = conf, label

                    now = time.time()
                    if state["last_logged"] != label or (now - state.get("last_log_time", 0) > 30):
                        timestamp = int(now)
                        img_name = f"history_{cam_id}_{label}_{timestamp}.jpg"
                        img_path = os.path.join(HISTORY_FOLDER, img_name)
                        cv2.imwrite(img_path, annotated_frame)
                        log_detection_event(label, f"history/{img_name}", "Live Detection", conf, camera_id=cam_id)
                        state["last_logged"] = label
                        state["last_log_time"] = now

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Scenario B: Hybrid Check (Low Confidence + Movement = Unknown)
                    elif 0.15 < conf <= 0.70:
                        is_moving_anomaly = False
                        for (mx1, my1, mx2, my2) in moving_boxes:
                            if mx1 < x2 and mx2 > x1 and my1 < y2 and my2 > y1:
                                is_moving_anomaly = True
                                break
                        
                        if is_moving_anomaly:
                            pest_found_in_this_frame = True
                            if state["ai_override"]:
                                best_pest = state["ai_override"]
                                display_text, color = f"AI: {best_pest}", (0, 255, 0)
                            else:
                                best_pest = "Unknown"
                                display_text, color = "Analyzing..." if state["ai_processing"] else "Unknown", (0, 0, 255)
                                
                                now = time.time()
                                if not state["ai_processing"] and (now - state["last_request"] > 3):
                                    state["last_request"] = now
                                    state["ai_processing"] = True
                                    threading.Thread(target=start_ai_analysis_thread, 
                                                     args=(frame.copy(), cam_id, x1, y1, x2, y2, frame_w, frame_h)).start()

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, display_text, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if pest_found_in_this_frame:
        state["detected_pest"] = best_pest
        state["confidence"] = best_conf
        state["timeout"] = time.time() + (5.0 if state["ai_override"] else 2.0)
        
        now = time.time()
        
        if best_pest and best_pest.lower() != "unknown":
            last_log_time = state.get("last_log_time", 0)
            if (now - last_log_time > 10.0):
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_filename = f"hist_{cam_id.replace(' ', '')}_{timestamp_str}.jpg"
                snap_path = os.path.join(HISTORY_FOLDER, snap_filename)
                cv2.imwrite(snap_path, annotated_frame) 
                
                log_detection_event(
                    pest_name=best_pest, 
                    image_path=f"history/{snap_filename}", 
                    detection_type="Local Model (YOLO)", 
                    camera_id=cam_id
                )
                
                state["last_logged"] = best_pest
                state["last_log_time"] = now 

    return annotated_frame

def generate_frames_cam1():
    camera = get_camera(0)
    while True:
        if camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success: break
            processed = process_camera_frame(frame, "CAM 1") if is_detection_running else frame
            ret, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Yield blank frame if camera not available
            blank = get_blank_frame("CAMERA 1 NOT FOUND")
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)  # Avoid busy loop

def generate_frames_cam2():
    camera = get_camera(1)
    while True:
        if camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success: break
            processed = process_camera_frame(frame, "CAM 2") if is_detection_running else frame
            ret, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Yield blank frame if camera not available
            blank = get_blank_frame("CAMERA 2 NOT FOUND")
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)  # Avoid busy loop

def generate_frames_cam3():
    camera = get_camera(2)
    while True:
        if camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success: break
            processed = process_camera_frame(frame, "CAM 3") if is_detection_running else frame
            ret, buffer = cv2.imencode('.jpg', processed)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Yield blank frame if camera not available
            blank = get_blank_frame("CAMERA 3 NOT FOUND")
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)  # Avoid busy loop

@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames_cam1(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_2')
def video_feed_2():
    return Response(generate_frames_cam2(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_3')
def video_feed_3():
    return Response(generate_frames_cam3(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== API ROUTES ==================

# --- ROUTE: Active Learning Maintenance Controls ---
@app.route('/api/maintenance/train-now', methods=['POST'])
@login_required 
def api_train_now():
    global is_detection_running
    
    # 1. Pause the cameras
    is_detection_running = False
    
    # 2. Launch the training script
    try:
        script_path = os.path.join(BASE_DIR, 'train_update.py')
        subprocess.Popen([sys.executable, script_path])
        return jsonify({'success': True, 'message': 'Background training started! Cameras paused.'})
    except Exception as e:
        is_detection_running = True
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/api/maintenance/pause', methods=['POST'])
def api_maintenance_pause():
    """Temporarily disables YOLO inference to free up CPU for background training."""
    global is_detection_running
    is_detection_running = False
    print("⏸️ SYSTEM PAUSED: CPU freed for background training.")
    return jsonify({'success': True, 'status': 'AI Inference Paused'})

@app.route('/api/maintenance/resume', methods=['POST'])
def api_maintenance_resume():
    """Hot-swaps the newly trained model and resumes live detection."""
    global is_detection_running, model, frame_lock
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    try:
        new_model_path = request.json.get('model_path', 'new_best.pt')  # type: ignore
        with frame_lock:
            if os.path.exists(new_model_path):
                print(f"🔄 Hot-swapping to updated model: {new_model_path}")
                model = YOLO(new_model_path)
            else:
                print("⚠️ New model not found. Resuming with current model.")
        
        is_detection_running = True
        
        # Reset cooldowns so it doesn't immediately log old pests
        for cam in cam_states:
            cam_states[cam]["last_logged"] = None 
            cam_states[cam]["last_log_time"] = 0 
            
        print("▶️ SYSTEM RESUMED: New brain loaded, live detection active.")
        return jsonify({"success": True, "message": "Resumed successfully"})
    except Exception as e:
        is_detection_running = True # Ensure it turns back on even if swap fails
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/start', methods=['POST'])
def api_start():
    global is_detection_running, cam_states, ACTIVE_USER_EMAIL
    if request.is_json and request.json and 'email' in request.json:
        ACTIVE_USER_EMAIL = request.json['email']
        session['user_email'] = request.json['email']
        
    is_detection_running = True
    for cam in cam_states:
        cam_states[cam]["last_logged"] = None 
        cam_states[cam]["last_log_time"] = 0  
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global is_detection_running, cam_states
    is_detection_running = False
    for cam in cam_states:
        cam_states[cam]["detected_pest"] = "" 
        cam_states[cam]["ai_override"] = None
        cam_states[cam]["ai_processing"] = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/api/status')
def api_status():
    global is_detection_running, cam_states
    
    if not is_detection_running:
        return jsonify({"running": False, "status_text": "Stopped"})

    response_data = {"running": True, "cameras": {}}
    
    try:
        conn = get_db()
        cur = conn.cursor()
        
        for cam_id in ["CAM 1", "CAM 2", "CAM 3"]:
            state = cam_states[cam_id]
            
            if state["detected_pest"] and time.time() > state["timeout"]:
                state["detected_pest"] = ""
                state["ai_override"] = None
                state["ai_cache"] = None

            current_name = state["ai_override"] or state["detected_pest"]
            
            cam_res = {
                "status_text": "Scanning...",
                "pest_name": "—", "scientific_name": "—", "classification": "—",
                "cultural": "—", "biological": "—", "sanitation": "—",
                "mechanical": "—", "chemical": "—", "pest_photo": None,
            }
            
            if current_name:
                if current_name == "Unknown":
                    if state["ai_processing"]:
                        cam_res.update({"status_text": "🤖 AI is Analyzing...", "pest_name": "Identifying..."})
                    elif state["ai_cache"]:
                        cam_res.update({
                            "status_text": f"AI Identified: {state['ai_cache']['common_name']}",
                            "pest_name": state['ai_cache']['common_name'],
                            "scientific_name": state['ai_cache'].get('scientific_name', 'N/A')
                        })
                    else:
                        cam_res['status_text'] = "Unknown Object Detected"
                else:
                    cur.execute("SELECT * FROM pests WHERE common_name = ? OR yolo_name = ? LIMIT 1", 
                                (current_name, current_name))
                    pest_info = cur.fetchone()
                    
                    if pest_info:
                        pest_dict = dict(pest_info)
                        display_name = pest_dict.get('common_name', current_name)
                        cam_res.update({
                            "status_text": f"Detected: {display_name}",
                            "pest_name": display_name,
                            "scientific_name": pest_dict.get('scientific_name', 'N/A'),
                            "classification": pest_dict.get('classification', 'N/A'),
                            "cultural": pest_dict.get('cultural_methods', '—'),
                            "biological": pest_dict.get('biological_control', '—'),
                            "sanitation": pest_dict.get('sanitation', '—'),
                            "mechanical": pest_dict.get('mechanical_control', '—'),
                            "chemical": pest_dict.get('chemical_control', '—'),
                            "pest_photo": url_for('static', filename=pest_dict.get('image')) if pest_dict.get('image') else None
                        })
                    elif state["ai_cache"]:
                            cam_res.update({
                                "status_text": f"AI Identified: {state['ai_cache']['common_name']}",
                                "pest_name": state['ai_cache']['common_name'],
                                "scientific_name": state['ai_cache'].get('scientific_name', 'N/A'),
                                "cultural": state['ai_cache'].get('cultural_methods', '—'),
                            })
                        
            response_data["cameras"][cam_id] = cam_res
            
        return jsonify(response_data)
    except Exception as e:
        print(f"Status Error: {e}")
        return jsonify({"running": is_detection_running, "error": str(e)})

# ================== WEB ROUTES ==================

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            return render_template('login.html', error="Please fill in all fields.")

        session_token = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()

        if username == 'Admin' and password == 'admin123':
            session['admin'] = username
            session['role'] = 'main'
            session['session_token'] = session_token
            session.permanent = True
            session['last_activity'] = time.time()
            return redirect(url_for('admin_dashboard'))

        conn = None 
        try:
            admin_db = os.path.join(DB_DIR, 'admin_db.db')
            conn = sqlite3.connect(admin_db)
            c = conn.cursor()
            c.execute("SELECT username, password, role FROM admins WHERE username=?", (username,))
            admin = c.fetchone()
            
            if admin and check_password_hash(admin[1], password):
                session['admin'] = admin[0]
                session['role'] = admin[2]
                session['session_token'] = session_token
                session.permanent = True
                session['last_activity'] = time.time()
                return redirect(url_for('admin_dashboard'))
            else:
                return render_template('login.html', error="Invalid username or password.")
        except Exception:
            return render_template('login.html', error="A server error occurred during login.")
        finally:
            if conn: conn.close() 

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/admin_dashboard')
@login_required
@restrict_url_access
def admin_dashboard():
    conn = None
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        c = conn.cursor()
        c.execute("SELECT * FROM pests")
        pests = c.fetchall()
        return render_template('admin_dashboard.html', pests=pests, admin_name=session.get('admin'))
    except Exception as e:
        print(f"Error fetching dashboard data: {e}")
        flash("Could not load pest data.", "danger")
        return redirect(url_for('home'))
    finally:
        if conn: conn.close()

@app.route('/add_pest', methods=['GET', 'POST'])
@login_required
@restrict_url_access
def add_pest():
    session.pop('_flashes', None)
    
    if request.method == 'POST':
        conn = None
        common_name = request.form.get('common_name', '').strip()
        image = request.files.get('image')

        if not common_name or not (image and image.filename):
            flash("Please ensure required fields and an image are selected.", "danger")
            return render_template('add_pest.html')

        try:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            image_filename_to_save = f'uploads/{filename}' 
            
            pest_db = os.path.join(DB_DIR, 'pests_add.db')
            conn = sqlite3.connect(pest_db)
            c = conn.cursor()
            
            c.execute('''INSERT INTO pests (type, common_name, scientific_name, order_name, family, classification,
                        cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image, yolo_name)
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        "Native Species",
                        common_name, request.form.get('scientific_name', ''), 
                        request.form.get('order_name', ''), request.form.get('family', ''), 
                        request.form.get('classification', ''),
                        request.form.get('cultural_methods', ''), request.form.get('biological_control', ''), 
                        request.form.get('sanitation', ''), request.form.get('mechanical_control', ''), 
                        request.form.get('chemical_control', ''), image_filename_to_save, common_name
            ))
            conn.commit()
            flash("Pest successfully registered!", "success")
            return redirect(url_for('pest_list'))
        except Exception as e:
            flash(f"Error adding pest: {e}", "danger")
        finally:
            if conn: conn.close()
                
    return render_template('add_pest.html')

@app.route('/delete_pest/<int:pest_id>', methods=['POST'])
@login_required
@restrict_url_access
def delete_pest(pest_id):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM pests WHERE id = ?", (pest_id,))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_pest_image', methods=['POST'])
@login_required
@restrict_url_access
def upload_pest_image():
    pest_id = request.form.get('pest_id')
    image_file = request.files.get('image_file')

    if not pest_id or not image_file or not image_file.filename:
        return jsonify({'success': False, 'error': 'Missing Pest ID or image file.'}), 400

    try:
        filename_base, file_ext = os.path.splitext(image_file.filename)
        safe_filename = secure_filename(f"pest_{pest_id}{file_ext}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        image_file.save(filepath)
        image_db_path = f"uploads/{safe_filename}" 
        
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE pests SET image = ? WHERE id = ?", (image_db_path, pest_id))
        conn.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Image updated successfully!', 
            'image_url': url_for('static', filename=image_db_path)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/update_pests', methods=['POST'])
@login_required
@restrict_url_access
def update_pests():
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Invalid request format.'}), 400

    try:
        data = request.get_json()
        pests_to_update = data.get('pests')
        
        conn = get_db()
        cur = conn.cursor()
        errors = []
        success_count = 0

        for pest in pests_to_update:
            try:
                cur.execute("""
                    UPDATE pests
                    SET type = ?, common_name = ?, scientific_name = ?, order_name = ?, family = ?, 
                        classification = ?, cultural_methods = ?, biological_control = ?, sanitation = ?, 
                        mechanical_control = ?, chemical_control = ?
                    WHERE id = ?
                """, (
                    pest.get('type'), pest.get('common_name'), pest.get('scientific_name'), pest.get('order_name'),
                    pest.get('family'), pest.get('classification'), pest.get('cultural_methods'), pest.get('biological_control'),
                    pest.get('sanitation'), pest.get('mechanical_control'), pest.get('chemical_control'),
                    pest.get('id')
                ))
                success_count += 1
            except Exception as e:
                errors.append(f"Failed to update pest ID {pest.get('id')}: {str(e)}")

        conn.commit()
        
        if errors:
            return jsonify({'success': False, 'message': f"Completed with errors. {success_count} updated.", 'errors': errors})
        else:
            return jsonify({'success': True, 'message': 'All changes saved successfully!'})

    except Exception:
        return jsonify({'success': False, 'error': 'A critical server error occurred.'}), 500

@app.route('/register', methods=['GET', 'POST'])
@login_required
@restrict_url_access
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']

        if password != confirm:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))

        conn = None 
        try:
            admin_db = os.path.join(DB_DIR, 'admin_db.db')
            conn = sqlite3.connect(admin_db)
            c = conn.cursor()
            c.execute("SELECT * FROM admins WHERE username=?", (username,))
            existing = c.fetchone()

            if existing:
                flash("Username already exists!", "danger")
                return redirect (url_for('register'))
            
            hashed = generate_password_hash(password)
            c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            flash("Admin succesfully registered!", "success")
            return redirect(url_for('register'))
        except Exception as e:
            flash(f"Error registering admin: {e}", "danger")
        finally:
            if conn: conn.close()

    return render_template('register.html')

@app.route('/pest_list')
@login_required
@restrict_url_access
def pest_list():
    conn = None 
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM pests")
        pests = cur.fetchall()
        return render_template('pest_list.html', pests=pests)
    except Exception as e:
        flash("Could not load pest list.", "danger")
        return redirect(url_for('home'))
    finally:
        if conn: conn.close()

@app.route('/upload', methods=['GET', 'POST'])
@restrict_url_access
def upload():
    image_url = None 

    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file or not file.filename:
            flash("No file selected.", "danger")
            return redirect(request.url)

        try:
            filename = secure_filename(str(file.filename)) 
            filepath = os.path.join(UPLOAD_FOLDER, filename) 
            file.save(filepath)
            
            image_url = url_for('static', filename='uploads/' + filename)
            
            detected_name = None
            
            img = cv2.imread(filepath)
            if img is not None:
                img_h, img_w, _ = img.shape
            else:
                img_h, img_w = 480, 640 

            if model:
                results = model(filepath, conf=0.25)
                if results and len(results) > 0 and results[0].boxes:
                    best_conf_index = results[0].boxes.conf.argmax()
                    class_index = int(results[0].boxes.cls[best_conf_index].item())
                    raw_label = results[0].names[class_index]
                    conf = float(results[0].boxes.conf[best_conf_index].item())
                    
                    # Apply mapping from the global dictionary
                    label = PEST_ALIASES.get(raw_label, raw_label)

                    box = results[0].boxes.xyxy[best_conf_index].tolist()
                    x1, y1, x2, y2 = box
                    
                    if conf > 0.60:
                        detected_name = label
                    elif 0.15 < conf <= 0.60:
                        detected_name = "Unknown"

            if detected_name == "Unknown" or detected_name is None:
                print("⚡ Triggering AI Analysis for Upload...")
                ai_data = fetch_pest_info_from_ai("Unknown", image_path=filepath)
                
                # --- NEW: HARDCODED SAFETY FILTER ---
                if ai_data:
                    c_name = str(ai_data.get('common_name', '')).lower()
                    cls_name = str(ai_data.get('classification', '')).lower()
                    forbidden_words = ['disease', 'wilt', 'rot', 'virus', 'fungus', 'plant', 'leaf', 'pineapple', 'crop']
                    
                    if any(word in c_name or word in cls_name for word in forbidden_words):
                        print(f"🚫 BLOCKED: AI attempted to log a plant/disease ({c_name}). Ignored.")
                        ai_data['common_name'] = "N/A" 
                
                if ai_data and ai_data.get('common_name') not in ["N/A", "Standard Name", None, "n/a"]:
                    detected_name = ai_data.get('common_name')
                    
                    with db_lock:
                    # ... (KEEP THE REST OF YOUR EXISTING CODE BELOW THIS UNCHANGED) ...
                        conn = get_db()
                        c = conn.cursor()
                        c.execute("SELECT id FROM pests WHERE common_name = ?", (detected_name,))
                        if not c.fetchone():
                            db_image_path = f"uploads/{filename}"
                            c.execute('''INSERT INTO pests (type, 
                                common_name, scientific_name, order_name, family, classification,
                                cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image, yolo_name
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                                "Non-Native Species", detected_name, 
                                ai_data.get('scientific_name', 'N/A'), ai_data.get('order_name', 'N/A'), 
                                ai_data.get('family', 'N/A'), ai_data.get('classification', 'Non-Native/Intruder'), 
                                ai_data.get('cultural_methods', '—'), ai_data.get('biological_control', '—'), 
                                ai_data.get('sanitation', '—'), ai_data.get('mechanical_control', '—'), 
                                ai_data.get('chemical_control', '—'), db_image_path, detected_name
                            ))
                            conn.commit()
                else:
                    flash("AI could not identify the subject.", "warning")
                    return render_template('pest_upload.html', image_url=image_url)

            if detected_name and detected_name.lower() != "unknown":
                formatted_name = detected_name.strip()
                log_detection_event(formatted_name, f"uploads/{filename}", "Manual Upload", conf, camera_id="Upload")
                conn = get_db()
                pest_info = conn.execute(
                    "SELECT * FROM pests WHERE common_name = ? COLLATE NOCASE OR yolo_name = ? COLLATE NOCASE",
                    (formatted_name, formatted_name)
                ).fetchone()

                if pest_info:
                    return render_template('pest_upload.html', pest=pest_info, image_url=image_url)
                else:
                    dummy_pest = {
                        "common_name": formatted_name,
                        "scientific_name": "Identified by AI",
                        "classification": "Non-Native/Intruder",
                        "cultural_methods": "—", "biological_control": "—",
                        "sanitation": "—", "mechanical_control": "—", "chemical_control": "—"
                    }
                    flash(f"Identified: {formatted_name}", "success")
                    return render_template('pest_upload.html', pest=dummy_pest, image_url=image_url)
            
            flash("No identifiable pest found.", "info")
            return render_template('pest_upload.html', image_url=image_url)
                
        except Exception as e:
            print(f"Upload Error: {e}")
            flash(f"Error processing image: {e}", "danger")
            return render_template('pest_upload.html', image_url=image_url)
            
    return render_template('pest_upload.html', image_url=image_url)

@app.route('/library')
@restrict_url_access
def pest_library():
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM pests")
        pests = cur.fetchall()
        return render_template('pest_library.html', pests=pests)
    except Exception:
        return redirect(url_for('home'))
    
@app.route('/history')
def detection_history():
    conn = None
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 500")
        history_logs = cur.fetchall()
        return render_template('detection_history.html', logs=history_logs)
    except Exception as e:
        print(f"Error loading history: {e}")
        flash("Could not load detection history.", "danger")
        return redirect(url_for('user_page'))
    finally:
        if conn: conn.close()

@app.route('/user')
def user_page(): return render_template('user.html') 

@app.route('/index')
@restrict_url_access
def index_page(): return render_template('index.html')

# ================== ACTIVE LEARNING SCHEDULER ==================

def trigger_background_training():
    """Pauses live detection and spawns the training subprocess."""
    global is_detection_running
    print("🕒 Scheduled AI Maintenance starting...")
    is_detection_running = False
    try:
        script_path = os.path.join(BASE_DIR, 'train_update.py')
        subprocess.Popen([sys.executable, script_path])
        print("🚀 Background training script launched successfully.")
    except Exception as e:
        print(f"❌ Failed to launch training script: {e}")
        is_detection_running = True

def check_if_training_needed_on_startup():
    TRACKER_FILE = os.path.join(DATASET_DIR, 'last_trained_count.json')
    
    if not os.path.exists(DATASET_LBL_DIR):
        return

    current_count = len([name for name in os.listdir(DATASET_LBL_DIR) if name.endswith('.txt')])
    
    # Load how many images we had during the last training
    last_count = 0
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            last_count = json.load(f).get('count', 0)

    # Calculate NEW images only
    new_images = current_count - last_count

    if new_images >= 30:
        print(f"⚠️ Startup Check: Found {new_images} NEW images. Initiating catch-up...")
        
        # Update the tracker file before starting
        with open(TRACKER_FILE, 'w') as f:
            json.dump({'count': current_count}, f)
            
        trigger_background_training()
    else:
        print(f"✅ Startup Check: Only {new_images} new images found. Skipping auto-train.")

scheduler = BackgroundScheduler(timezone="Asia/Manila")
scheduler.add_job(func=trigger_background_training, trigger="cron", hour=2, minute=0)
scheduler.start()
check_if_training_needed_on_startup()

if __name__ == '__main__':
    def release_cameras():
        global cams
        for i in cams:
            cap = cams[i]  
            if cap is not None and cap.isOpened():
                cap.release()
        print("Cameras released.")
        
    atexit.register(release_cameras)
    app.run(debug=True, use_reloader=False, threaded=True)