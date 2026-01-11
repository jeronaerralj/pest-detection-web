# app.py imports
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify, Response
import sqlite3, os, cv2, threading, atexit, re, datetime, time, json
import numpy as np 
import base64 
from ultralytics.models.yolo import YOLO 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import google.generativeai as genai 
from groq import Groq 
# --- NEW IMPORTS ---
from openai import OpenAI  # For GitHub Models
import ollama              # For Local Llama
# -------------------
import PIL.Image 
from dotenv import load_dotenv
from functools import wraps
from datetime import timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_fallback_key')
app.permanent_session_lifetime = timedelta(seconds=int(os.getenv('SESSION_TIMEOUT_SEC', '900')))

# ================== AI CONFIGURATION ==================
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY') 
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN') # <--- NEW TOKEN

if not GENAI_API_KEY:
    print("⚠️ WARNING: GENAI_API_KEY not found in .env file")
else:
    try:
        genai.configure(api_key=GENAI_API_KEY)
        print("✅ Gemini AI Configured Successfully")
    except Exception as e:
        print(f"Error configuring Gemini: {e}")

if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found (Text fallback disabled)")

if not GITHUB_TOKEN:
    print("⚠️ WARNING: GITHUB_TOKEN not found (GitHub Vision fallback disabled)")

# ================== GLOBAL STATE & LOCKS ==================
frame_lock = threading.Lock()
db_lock = threading.Lock()

last_detected_pest = ""           
is_detection_running = False      
last_annotated_frame = None       
last_confidence = 0.0
last_logged_pest = None 

# AI Threading State
is_ai_processing = False
ai_cooldown_timer = 0
# --- FIX: INCREASED COOLDOWN ---
AI_COOLDOWN_SECONDS = 60  # Increased from 15 to 60 to prevent Gemini Quota errors

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, 'database')
DATABASE = os.path.join(DB_DIR, 'pests_add.db') 

STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
HISTORY_FOLDER = os.path.join(STATIC_FOLDER, 'history') 

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True) 

# ================== HELPERS ==================

def clean_json_text(text):
    """Aggressively cleans JSON text returned by LLMs."""
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

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        is_api = request.path.startswith('/api') or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        if 'admin' not in session:
            if is_api:
                return jsonify({'success': False, 'error': 'Authentication required'}), 403
            flash("Please log in to access that page.", "warning")
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
    # Data structure we expect back
    json_structure = {
        "common_name": "Standard Name",
        "scientific_name": "Latin Name",
        "classification": "Insect/Fungi/etc",
        "family": "Family Name",
        "order_name": "Order Name",
        "cultural_methods": "1-2 sentence advice",
        "biological_control": "1-2 sentence advice",
        "sanitation": "1-2 sentence advice",
        "mechanical_control": "1-2 sentence advice",
        "chemical_control": "1-2 sentence advice"
    }

    # --- SCENARIO A: VISION IDENTIFICATION ---
    if (pest_name.lower() in ["unknown", "negative"]) and image_path:
        print(f"👁️ AI Vision: Analyzing image...")
        
        # 1. Try Gemini Vision (Primary)
        if GENAI_API_KEY:
            gemini_candidates = ['gemini-2.5-flash', 'gemini-exp-1206', 'gemini-flash-latest']
            try:
                img = PIL.Image.open(image_path)
                vision_prompt = f"Analyze this image. Identify the specific pest. Return JSON: {json.dumps(json_structure)}. If uncertain, return common_name: N/A."

                for model_name in gemini_candidates:
                    try:
                        print(f"   ...Trying Gemini: {model_name}")
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content([vision_prompt, img])
                        return json.loads(clean_json_text(response.text))
                    except Exception as e:
                        if "429" in str(e) or "quota" in str(e).lower():
                            print(f"   ⚠️ Quota Hit on {model_name}.")
                            continue 
                        else:
                            print(f"   ❌ Error with {model_name}: {e}")
            except Exception as e:
                print(f"❌ Gemini Vision Critical Fail: {e}")

        # 2. Try GitHub Models (Option 1 - Cloud Backup)
        if GITHUB_TOKEN:
            print("   👉 Switching to GitHub Models (Llama 3.2 Vision)...")
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                client = OpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=GITHUB_TOKEN
                )

                response = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Identify pest. Return JSON: {json.dumps(json_structure)}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }],
                    model="Llama-3.2-90B-Vision-Instruct",
                    temperature=0,
                )
                return json.loads(clean_json_text(response.choices[0].message.content))
            except Exception as e:
                print(f"❌ GitHub Models failed: {e}")

        # 3. Try Ollama (Option 3 - Local Backup)
        print("   👉 Switching to Local Ollama...")
        try:
            response = ollama.chat(
                model='llama3.2-vision',
                messages=[{
                    'role': 'user',
                    'content': f"Identify pest. Return JSON: {json.dumps(json_structure)}",
                    'images': [image_path]
                }]
            )
            return json.loads(clean_json_text(response['message']['content']))
        except Exception as e:
            print(f"❌ Ollama failed (Is it running?): {e}")

        print("❌ All AI Vision models failed.")
        return None

    # --- SCENARIO B: TEXT LOOKUP ---
    system_prompt = f"Expert Pineapple agronomy. Details for '{pest_name}'. JSON format: {json.dumps(json_structure)}."

    # 1. Try Gemini Text
    if GENAI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(system_prompt)
            return json.loads(clean_json_text(response.text))
        except: pass

    # 2. Try Groq Text (Legacy/Fallback)
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Output JSON only."},
                    {"role": "user", "content": system_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0,
                response_format={"type": "json_object"} 
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception: pass
        
    return None

def process_unknown_pest_background(image_path):
    """Background thread function to handle AI analysis."""
    global is_ai_processing, last_detected_pest
    
    print("🚀 Background Thread Started: Analyzing Unknown Pest...")
    try:
        # Call the AI logic
        ai_data = fetch_pest_info_from_ai("Unknown", image_path=image_path)

        if ai_data and ai_data.get('common_name') not in ["N/A", "Standard Name", None]:
            identified_name = ai_data.get('common_name')
            print(f"✅ AI Identified: {identified_name}")

            # Save to Database
            with db_lock:
                conn = sqlite3.connect(DATABASE, timeout=10)
                c = conn.cursor()
                
                # Check duplicate
                c.execute("SELECT id FROM pests WHERE common_name = ?", (identified_name,))
                if not c.fetchone():
                    # Use the filename from the temp path
                    filename = os.path.basename(image_path)
                    db_image_path = f"uploads/{filename}"
                    
                    c.execute('''INSERT INTO pests (
                        crop, common_name, scientific_name, order_name, family, classification,
                        cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image, yolo_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        "Pineapple", 
                        identified_name, 
                        ai_data.get('scientific_name'), 
                        ai_data.get('order_name'), 
                        ai_data.get('family'), 
                        ai_data.get('classification'), 
                        ai_data.get('cultural_methods'), 
                        ai_data.get('biological_control'), 
                        ai_data.get('sanitation'), 
                        ai_data.get('mechanical_control'), 
                        ai_data.get('chemical_control'), 
                        db_image_path,
                        identified_name 
                    ))
                    conn.commit()
                    print(f"💾 Saved {identified_name} to Database.")
                conn.close()

            # Update the global variable so UI sees the new name immediately
            last_detected_pest = identified_name
        else:
            print("❌ AI returned N/A or failed to identify.")
            
    except Exception as e:
        print(f"❌ Background Thread Error: {e}")
    finally:
        is_ai_processing = False  # Release the lock

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
            crop TEXT,
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
            detection_type TEXT      
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
# 1. Custom Pest Model
try:
    model = YOLO('datapest.pt')
    print("✅ Custom Pest Model Loaded")
except:
    print("⚠️ WARNING: datapest.pt not found. Detection will fail.")
    model = None

# 2. General Model (The "Double Agent" for Birds/Animals)
try:
    general_model = YOLO('yolov8n.pt')
    print("✅ General Model Loaded (for Intruder Detection)")
except Exception as e:
    print(f"⚠️ General model failed to load: {e}")
    general_model = None

# COCO Dataset IDs for animals we want to treat as "Unknown/Intruders"
ANIMAL_CLASSES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# ================== LOGGING & CAMERA ==================
def log_detection_event(pest_name, image_path, detection_type):
    with db_lock:
        try:
            conn = sqlite3.connect(DATABASE, timeout=10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                user = session.get('admin', 'SYSTEM') 
            except:
                user = 'SYSTEM'
            image_path = image_path.replace('\\', '/')
            conn.execute("""
                INSERT INTO history (timestamp, yolo_name, image_path, user_session, detection_type)
                VALUES (?, ?, ?, ?, ?)
            """, (current_time, pest_name, image_path, user, detection_type))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging history: {e}")

def handle_continuous_logging(pest_name):
    global last_logged_pest, last_annotated_frame
    
    if pest_name and pest_name != last_logged_pest:
        frame_copy = None
        with frame_lock:
            if last_annotated_frame is not None:
                frame_copy = last_annotated_frame.copy()
        
        if frame_copy is not None:
            try:
                safe_name = secure_filename(f"{pest_name}_{int(time.time())}.jpg")
                save_path = os.path.join(STATIC_FOLDER, 'history', safe_name)
                cv2.imwrite(save_path, frame_copy)
                
                db_path = f"history/{safe_name}"
                log_detection_event(pest_name, db_path, 'Continuous Feed')
                
                last_logged_pest = pest_name 
                print(f"✅ Auto-logged: {pest_name}")
            except Exception as e:
                print(f"Error continuous logging: {e}")

cams = {0: None, 1: None, 2: None}

def get_camera(index):
    global cams
    if cams[index] is None or not cams[index].isOpened():
        cams[index] = cv2.VideoCapture(index, cv2.CAP_DSHOW) 
    return cams[index]

def get_blank_frame(text="CAMERA NOT FOUND"):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', blank)
    return buffer.tobytes()

def generate_frames_cam1():
    global last_detected_pest, is_detection_running, last_annotated_frame, last_confidence
    camera = get_camera(0)

    while True:
        if not camera.isOpened():
            camera = get_camera(0)
            if not camera.isOpened():
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 1 ERROR") + b'\r\n')
                time.sleep(2)
                continue

        success, frame = camera.read()
        if not success:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 1 NO SIGNAL") + b'\r\n')
            time.sleep(0.5)
            continue
        
        annotated_frame = frame
        
        if is_detection_running:
            # --- STEP 1: Run Custom Model (Priority) ---
            pest_found = False
            if model:
                results = model(frame, stream=True, conf=0.50, verbose=False, agnostic_nms=True)
                
                best_conf = 0.0
                best_pest = None

                for r in results:
                    if len(r.boxes) > 0:
                        annotated_frame = r.plot()
                        pest_found = True
                        for box in r.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            label = r.names[cls_id]

                            if label.lower() in ["negative", "alienated"]:
                                label = "Unknown"

                            if conf > best_conf:
                                best_conf = conf
                                best_pest = label
                
                if pest_found and best_pest:
                    with frame_lock:
                        last_annotated_frame = annotated_frame
                    last_detected_pest = best_pest
                    last_confidence = best_conf

            # --- STEP 2: Run General Model (If no pest found) ---
            if not pest_found and general_model:
                gen_results = general_model(frame, classes=ANIMAL_CLASSES, conf=0.40, verbose=False)
                for gr in gen_results:
                    if len(gr.boxes) > 0:
                        annotated_frame = gr.plot()
                        with frame_lock:
                            last_annotated_frame = annotated_frame
                        
                        last_detected_pest = "Unknown"
                        detected_animal = gr.names[int(gr.boxes[0].cls[0])]
                        print(f"⚠️ Intruder Detected: {detected_animal} -> Marking as Unknown")
            
            if not pest_found and (not general_model or len(gen_results[0].boxes) == 0):
                 with frame_lock:
                    last_annotated_frame = frame
        else:
            with frame_lock:
                last_annotated_frame = frame

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n') 

def generate_frames_cam2():
    camera = get_camera(1)
    while True:
        if camera is None or not camera.isOpened():
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 2 NOT FOUND") + b'\r\n')
            time.sleep(10)
            camera = get_camera(1)
            continue
        success, frame = camera.read()
        if not success: 
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 2 NO SIGNAL") + b'\r\n')
            time.sleep(0.5)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_frames_cam3():
    camera = get_camera(2)
    while True:
        if camera is None or not camera.isOpened():
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 3 NOT FOUND") + b'\r\n')
            time.sleep(10)
            camera = get_camera(2)
            continue
        success, frame = camera.read()
        if not success: 
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 3 NO SIGNAL") + b'\r\n')
            time.sleep(0.5)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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

@app.route('/api/start', methods=['POST'])
def api_start():
    global is_detection_running, last_logged_pest
    is_detection_running = True
    last_logged_pest = None 
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global is_detection_running, last_detected_pest
    is_detection_running = False
    last_detected_pest = "" 
    return jsonify({'status': 'Detection stopped'})

@app.route('/api/status')
def api_status():
    global last_detected_pest, is_detection_running, is_ai_processing, ai_cooldown_timer
    
    pest_name = last_detected_pest.strip()
    
    response_data = {
        "running": is_detection_running,
        "status_text": "Stopped" if not is_detection_running else "Scanning...",
        "pest_name": "—", "scientific_name": "—", "classification": "—",
        "cultural": "—", "biological": "—", "sanitation": "—",
        "mechanical": "—", "chemical": "—", "pest_photo": None,
    }

    if not is_detection_running:
        return jsonify(response_data)

    if not pest_name:
        response_data['status_text'] = "Scanning... (No pests detected)"
        return jsonify(response_data)

    # === NEW: THREADED UNKNOWN PROCESSING ===
    if pest_name == "Unknown":
        if is_ai_processing:
            response_data['status_text'] = "🤖 AI is Analyzing... (Please Wait)"
            response_data['pest_name'] = "Identifying..."
            return jsonify(response_data)
        
        # Check Cooldown
        if time.time() < ai_cooldown_timer:
             response_data['status_text'] = "Unknown Object Detected (Cooldown)"
             response_data['pest_name'] = "Unknown"
             return jsonify(response_data)

        # Trigger Background Analysis
        is_ai_processing = True
        ai_cooldown_timer = time.time() + AI_COOLDOWN_SECONDS
        
        frame_copy = None
        with frame_lock:
            if last_annotated_frame is not None:
                frame_copy = last_annotated_frame.copy()
        
        if frame_copy is not None:
            temp_filename = f"unknown_{int(time.time())}.jpg"
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            cv2.imwrite(temp_path, frame_copy)
            
            # Start Thread
            thread = threading.Thread(target=process_unknown_pest_background, args=(temp_path,))
            thread.daemon = True 
            thread.start()
            
            response_data['status_text'] = "Sending to AI..."
            response_data['pest_name'] = "Identifying..."
        
        return jsonify(response_data)
    # ========================================

    # STANDARD DB LOOKUP
    try:
        conn = get_db()
        cur = conn.cursor()
        pest_info = None
        
        # 1. Exact YOLO Name
        try:
            cur.execute("SELECT * FROM pests WHERE yolo_name = ? COLLATE NOCASE LIMIT 1", (pest_name,))
            pest_info = cur.fetchone()
        except: pass 

        # 2. Common Name
        if not pest_info:
            formatted_name = pest_name.replace('-', ' ').replace('_', ' ').title()
            cur.execute("SELECT * FROM pests WHERE common_name LIKE ? LIMIT 1", (formatted_name,))
            pest_info = cur.fetchone()

        # 3. Partial Match
        if not pest_info:
             cur.execute("SELECT * FROM pests WHERE common_name LIKE ? LIMIT 1", (f"%{pest_name}%",))
             pest_info = cur.fetchone()

        if pest_info:
            pest_dict = dict(pest_info)
            display_name = pest_dict.get('common_name', pest_name).strip()
            
            handle_continuous_logging(display_name)
            
            response_data.update({
                "status_text": f"Detected: {display_name}",
                "pest_name": display_name,
                "scientific_name": pest_dict.get('scientific_name', 'N/A'),
                "classification": pest_dict.get('classification', 'N/A'),
                "cultural": pest_dict.get('cultural_methods', 'N/A'),
                "biological": pest_dict.get('biological_control', 'N/A'),
                "sanitation": pest_dict.get('sanitation', 'N/A'),
                "mechanical": pest_dict.get('mechanical_control', 'N/A'),
                "chemical": pest_dict.get('chemical_control', 'N/A'),
                "pest_photo": url_for('static', filename=pest_dict.get('image')) if pest_dict.get('image') else None
            })
            return jsonify(response_data)
            
    except Exception as e:
        print(f"❌ Database Error: {e}")

    response_data['status_text'] = f"Detected '{pest_name}' (Not in DB)"
    response_data['pest_name'] = pest_name
    return jsonify(response_data)

# ================== WEB ROUTES ==================

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if username == 'Admin' and password == 'admin123':
            session['admin'] = username
            session['role'] = 'main'
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
def add_pest():
    session.pop('_flashes', None)
    
    if request.method == 'POST':
        conn = None
        crop_type = request.form.get('crop_type', '').strip()
        common_name = request.form.get('common_name', '').strip()
        image = request.files.get('image')

        if not crop_type or not common_name or not (image and image.filename):
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
            
            c.execute('''INSERT INTO pests (crop, common_name, scientific_name, order_name, family, classification,
                        cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image, yolo_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (crop_type, common_name, request.form.get('scientific_name', ''), 
                         request.form.get('order_name', ''), request.form.get('family', ''), 
                         request.form.get('classification', ''),
                         request.form.get('cultural_methods', ''), request.form.get('biological_control', ''), 
                         request.form.get('sanitation', ''), request.form.get('mechanical_control', ''), 
                         request.form.get('chemical_control', ''), image_filename_to_save, common_name))
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
                    SET crop = ?, common_name = ?, scientific_name = ?, order_name = ?, family = ?, 
                        classification = ?, cultural_methods = ?, biological_control = ?, sanitation = ?, 
                        mechanical_control = ?, chemical_control = ?
                    WHERE id = ?
                """, (
                    pest.get('crop'), pest.get('common_name'), pest.get('scientific_name'), pest.get('order_name'),
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
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        image_url = None
        
        if not file or not file.filename:
            flash("No file selected for uploading.", "danger")
            return redirect(request.url)

        try:
            filename = secure_filename(str(file.filename)) 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('static', filename='uploads/' + filename)
            
            if model:
                results = model(filepath)
                if results and len(results) > 0 and results[0].boxes and len(results[0].boxes.cls) > 0:
                    best_conf_index = results[0].boxes.conf.argmax()
                    class_index = int(results[0].boxes.cls[best_conf_index].item())
                    detected_pest_name = results[0].names[class_index]
                    
                    if detected_pest_name.lower() in ["negative", "alienated"]:
                        # For uploads, we can optionally add the AI lookup here too if you want,
                        # but keeping it simple for now as per your original code.
                        flash("Unknown pest detected. (Manual Upload AI analysis not yet enabled)", "warning")
                    else:
                        processed_name = detected_pest_name.strip().replace('-', ' ').replace('_', ' ').title()
                        conn = get_db() 
                        pest_info = conn.execute(
                            "SELECT * FROM pests WHERE common_name = ? COLLATE NOCASE",
                            (processed_name,)
                        ).fetchone()

                        if pest_info:
                            return render_template('pest_upload.html', pest=pest_info, image_url=image_url)
                        else:
                            flash(f"Pest '{detected_pest_name}' detected, but no DB info found.", "warning")
                else:
                    flash("No pest was detected in the uploaded image.", "info")
            return render_template('pest_upload.html', image_url=image_url)
                
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "danger")
            return render_template('pest_upload.html', image_url=image_url)
            
    return render_template('pest_upload.html')

@app.route('/library')
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

@app.route('/user')
def user_page(): return render_template('user.html') 

@app.route('/index')
def index_page(): return render_template('index.html')

if __name__ == '__main__':
    def release_cameras():
        global cams
        for i in cams:
            if cams[i] and cams[i].isOpened():
                cams[i].release()
        print("Cameras released.")
    
    atexit.register(release_cameras)
    app.run(debug=True, use_reloader=False, threaded=True)