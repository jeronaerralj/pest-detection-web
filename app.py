from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify, Response
import sqlite3, os, cv2
import numpy as np 
from ultralytics.models.yolo import YOLO 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import atexit 
import re 
import datetime 
import time
import json 
# UPDATED: Import the new SDK
from google import genai 
from groq import Groq 
import PIL.Image 
from dotenv import load_dotenv
from functools import wraps
from datetime import timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_fallback_key')
# Configure session timeout (seconds) and make sessions permanent so lifetime applies
app.permanent_session_lifetime = timedelta(seconds=int(os.getenv('SESSION_TIMEOUT_SEC', '900')))

# ================== AI CONFIGURATION ==================
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
if not GENAI_API_KEY:
    print("⚠️ WARNING: GENAI_API_KEY not found in .env file")

# Initialize the new Gemini Client
try:
    gemini_client = genai.Client(api_key=GENAI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    gemini_client = None

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def clean_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        newline_index = text.find('\n')
        if newline_index != -1:
            text = text[newline_index+1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def fetch_pest_info_from_ai(pest_name, image_path=None):
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

    # --- SCENARIO A: VISION IDENTIFICATION (Unknown/Negative) ---
    if pest_name.lower() == "negative" and image_path and gemini_client:
        print(f"👁️ AI Vision: Analyzing image to identify unknown organism...")
        try:
            img = PIL.Image.open(image_path)
            vision_prompt = f"""
            Analyze this image. Identify the specific agricultural pest or insect present on the plant.
            If no pest is clearly visible, return a JSON with "common_name": "N/A".
            Otherwise, provide details in strict JSON format matching: {json.dumps(json_structure)}.
            """
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[vision_prompt, img]
            )
            cleaned_text = clean_json_text(response.text)
            return json.loads(cleaned_text)
        except Exception as e_vision:
            print(f"❌ Vision Analysis Failed: {e_vision}")
            return None

    # --- SCENARIO B: TEXT LOOKUP (Known Name) ---
    system_prompt = f"""
    You are an agricultural expert. I have detected a pest named '{pest_name}' on a Pineapple crop.
    Provide details in strict JSON format matching this structure: {json.dumps(json_structure)}.
    Do not add conversational text. If data is unknown, use "N/A".
    """

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=system_prompt
            )
            cleaned_text = clean_json_text(response.text)
            return json.loads(cleaned_text)
        except Exception as e_gemini:
             print(f"❌ Gemini Text Failed: {e_gemini}")

    # Fallback to Groq
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful agricultural AI assistant that outputs only valid JSON object."},
                {"role": "user", "content": system_prompt}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0,
            response_format={"type": "json_object"} 
        )
        groq_response = chat_completion.choices[0].message.content
        return json.loads(groq_response)
    except Exception as e_groq:
        print(f"❌ Groq also failed: {e_groq}")
        return None

# ================== GLOBAL STATE & DATABASE SETUP ==================

last_detected_pest = ""           
is_detection_running = False      
last_annotated_frame = None       
last_confidence = 0.0

# Track what we last logged to avoid spamming DB in continuous mode
last_logged_pest = None 

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_DIR = os.path.join(BASE_DIR, 'database')
DATABASE = os.path.join(DB_DIR, 'pests_add.db') 

STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
HISTORY_FOLDER = os.path.join(STATIC_FOLDER, 'history') 

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True) 

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def login_required(f):
    """Require an active admin session and enforce session timeout.
    Returns JSON 403 for API or XHR requests, or redirects to login for browsers."""
    @wraps(f)
    def decorated(*args, **kwargs):
        is_api = request.path.startswith('/api') or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', '')
        if 'admin' not in session:
            if is_api:
                return jsonify({'success': False, 'error': 'Authentication required'}), 403
            flash("Please log in to access that page.", "warning")
            return redirect(url_for('login'))
        # Check session timeout
        last = session.get('last_activity')
        timeout = int(os.getenv('SESSION_TIMEOUT_SEC', '900'))
        now = time.time()
        if last and (now - last) > timeout:
            session.clear()
            if is_api:
                return jsonify({'success': False, 'error': 'Session expired'}), 403
            flash("Your session has expired. Please log in again.", "warning")
            return redirect(url_for('login'))
        # update activity
        session['last_activity'] = now
        return f(*args, **kwargs)
    return decorated

# --- MODEL LOADING ---
model = YOLO('Rhino.pt') 

def log_detection_event(pest_name, image_path, detection_type):
    conn = get_db()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = session.get('admin', 'SYSTEM') 
    try:
        conn.execute("""
            INSERT INTO history (timestamp, yolo_name, image_path, user_session, detection_type)
            VALUES (?, ?, ?, ?, ?)
        """, (current_time, pest_name, image_path, user, detection_type))
        conn.commit()
    except Exception as e:
        print(f"Error logging history: {e}")

# HELPER: Save image and log without stopping the stream
def handle_continuous_logging(pest_name):
    global last_logged_pest
    global last_annotated_frame
    
    # Only log if it's a NEW detection (different from the last one we saved)
    if pest_name and pest_name != last_logged_pest and last_annotated_frame is not None:
        try:
            safe_name = secure_filename(f"{pest_name}_{int(time.time())}.jpg")
            relative_path = os.path.join('history', safe_name)
            save_path = os.path.join(STATIC_FOLDER, relative_path)
            cv2.imwrite(save_path, last_annotated_frame)
            
            log_detection_event(pest_name, relative_path, 'Continuous Feed')
            
            # Update tracker so we don't save the same pest 100 times in 1 second
            last_logged_pest = pest_name 
            print(f"✅ Auto-logged new pest: {pest_name}")
        except Exception as e:
            print(f"Error in continuous logging: {e}")

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

# ================== CAMERA FEED (MULTI-CAM) ==================

# Initialize cameras. 0 is main, 1 and 2 are secondary.
cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1) 
cam3 = cv2.VideoCapture(2) 

def get_blank_frame(text="CAMERA NOT FOUND"):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', blank)
    return buffer.tobytes()

# --- CAM 1: MAIN CAMERA ---
def generate_frames_cam1():
    global last_detected_pest
    global is_detection_running 
    global last_annotated_frame 
    global last_confidence 
    
    while True:
        if not cam1.isOpened():
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 1 ERROR") + b'\r\n')
            time.sleep(1)
            continue

        success, frame = cam1.read()
        if not success:
            frame = last_annotated_frame if last_annotated_frame is not None else frame
            if frame is None: 
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 1 NO SIGNAL") + b'\r\n')
                continue
        
        annotated_frame = frame
        found_pest = ""
        current_conf = 0.0 
        
        if is_detection_running:
            results = model(frame, stream=True, conf=0.5, verbose=False) 
            for r in results:
                annotated_frame = r.plot()
                last_annotated_frame = annotated_frame 
                
                if r.boxes and len(r.boxes.cls) > 0:
                    best_conf_index = r.boxes.conf.argmax()
                    class_index = int(r.boxes.cls[best_conf_index].item())
                    found_pest = r.names[class_index]
                    current_conf = r.boxes.conf[best_conf_index].item()
                    print(f"👀 Model sees (Cam 1): {found_pest} ({current_conf:.2f})")
                    break 
            last_detected_pest = found_pest
            last_confidence = current_conf 
        else:
            annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
            last_detected_pest = ""
            last_confidence = 0.0

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- CAM 2 ---
def generate_frames_cam2():
    while True:
        if not cam2.isOpened():
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 2 NOT FOUND") + b'\r\n')
            time.sleep(2)
            continue
        success, frame = cam2.read()
        if not success: 
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 2 NO SIGNAL") + b'\r\n')
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- CAM 3 ---
def generate_frames_cam3():
    while True:
        if not cam3.isOpened():
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 3 NOT FOUND") + b'\r\n')
            time.sleep(2)
            continue
        success, frame = cam3.read()
        if not success: 
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + get_blank_frame("CAM 3 NO SIGNAL") + b'\r\n')
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed_1')
def video_feed_1() -> Response:
    return Response(generate_frames_cam1(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_2')
def video_feed_2() -> Response:
    return Response(generate_frames_cam2(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_3')
def video_feed_3() -> Response:
    return Response(generate_frames_cam3(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== API CONTROL ROUTES ==================

@app.route('/api/start', methods=['POST'])
@login_required
def api_start():
    global is_detection_running
    global last_logged_pest
    is_detection_running = True
    last_logged_pest = None # Reset log tracker
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
@login_required
def api_stop():
    global is_detection_running
    is_detection_running = False
    global last_detected_pest
    global last_annotated_frame
    
    last_detected_pest = "" 
    last_annotated_frame = None 
    return jsonify({'status': 'Detection stopped'})

@app.route('/api/status')
def api_status():
    global last_detected_pest
    global is_detection_running
    global last_annotated_frame
    global last_confidence 
    
    pest_name = last_detected_pest.strip()
    
    response_data = {
        "running": is_detection_running,
        "status_text": "Stopped" if not is_detection_running else "Detecting...",
        "pest_name": "—", 
        "scientific_name": "—",
        "classification": "—",
        "cultural": "—",
        "biological": "—",
        "sanitation": "—",
        "mechanical": "—",
        "chemical": "—",
        "pest_photo": None,
    }

    if not is_detection_running:
        return jsonify(response_data)

    # === LOGIC: HANDLE "NEGATIVE" (UNKNOWN) ===
    if pest_name.lower() == "unknown":
        if last_confidence <= 0.3:
             response_data['status_text'] = f"Scanning... (Background: {last_confidence:.2f})"
             return jsonify(response_data)
        
        response_data['status_text'] = "Analyzing Unknown Object..."
        
        saved_image_db_path = ""
        saved_abs_path = ""
        if last_annotated_frame is not None:
            filename = secure_filename(f"unknown_{int(time.time())}.jpg")
            saved_abs_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(saved_abs_path, last_annotated_frame)
            saved_image_db_path = f"uploads/{filename}"

        ai_data = fetch_pest_info_from_ai("Negative", image_path=saved_abs_path)

        if ai_data and ai_data.get('common_name') and ai_data.get('common_name') not in ["N/A", "Negative"]:
            new_name = ai_data.get('common_name')
            # Save to DB
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute('''INSERT INTO pests (
                    yolo_name, crop, common_name, scientific_name, order_name, family, 
                    classification, cultural_methods, biological_control, sanitation, 
                    mechanical_control, chemical_control, image
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    new_name, "Pineapple", new_name,
                    ai_data.get('scientific_name', 'N/A'), ai_data.get('order_name', 'N/A'),
                    ai_data.get('family', 'N/A'), ai_data.get('classification', 'N/A'),
                    ai_data.get('cultural_methods', 'N/A'), ai_data.get('biological_control', 'N/A'),
                    ai_data.get('sanitation', 'N/A'), ai_data.get('mechanical_control', 'N/A'),
                    ai_data.get('chemical_control', 'N/A'), saved_image_db_path
                ))
                conn.commit()
            except Exception as e:
                print(f"DB Error: {e}")

            # Auto-log continuous
            handle_continuous_logging(new_name)

            response_data.update({
                "status_text": f"Identified: {new_name}",
                "pest_name": new_name,
                "scientific_name": ai_data.get('scientific_name'),
                "classification": ai_data.get('classification'),
                "cultural": ai_data.get('cultural_methods'),
                "biological": ai_data.get('biological_control'),
                "sanitation": ai_data.get('sanitation'),
                "mechanical": ai_data.get('mechanical_control'),
                "chemical": ai_data.get('chemical_control'),
                "pest_photo": url_for('static', filename=saved_image_db_path)
            })
            return jsonify(response_data)
        else:
            response_data['status_text'] = "Analysis Failed: No pest identified."
            response_data['pest_name'] = "—" 
            return jsonify(response_data)

    elif pest_name == "":
        response_data['status_text'] = "Scanning... (No pests detected)"
        return jsonify(response_data)
    
    # === LOGIC: KNOWN PESTS ===
    pest_info = None
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM pests WHERE yolo_name = ? COLLATE NOCASE LIMIT 1", (pest_name,))
        pest_info = cur.fetchone() 

        if pest_info:
            pest_dict = dict(pest_info)
            display_name = pest_dict.get('common_name', pest_name).strip() 
            image_path = pest_dict.get('image')
            
            # Auto-log continuous
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
                "pest_photo": url_for('static', filename=image_path) if image_path else None
            })

        else:
            # KNOWN CLASS NOT IN DB -> TEXT AI
            saved_image_db_path = ""
            if last_annotated_frame is not None:
                filename = secure_filename(f"auto_trained_{pest_name}_{int(time.time())}.jpg")
                abs_path = os.path.join(UPLOAD_FOLDER, filename)
                cv2.imwrite(abs_path, last_annotated_frame)
                saved_image_db_path = f"uploads/{filename}"

            ai_data = fetch_pest_info_from_ai(pest_name)
            
            if ai_data:
                try:
                    cur.execute('''INSERT INTO pests (yolo_name, crop, common_name, scientific_name, order_name, family, 
                        classification, cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        pest_name, "Pineapple", ai_data.get('common_name', pest_name), ai_data.get('scientific_name', 'N/A'),
                        ai_data.get('order_name', 'N/A'), ai_data.get('family', 'N/A'), ai_data.get('classification', 'N/A'),
                        ai_data.get('cultural_methods', 'N/A'), ai_data.get('biological_control', 'N/A'), ai_data.get('sanitation', 'N/A'),
                        ai_data.get('mechanical_control', 'N/A'), ai_data.get('chemical_control', 'N/A'), saved_image_db_path
                    ))
                    conn.commit()
                except Exception as e_db:
                    print(f"Error saving to DB: {e_db}")

                final_name = ai_data.get('common_name', pest_name)
                handle_continuous_logging(final_name)

                response_data.update({
                    "status_text": f"AI Analyzed: {final_name}",
                    "pest_name": final_name,
                    "scientific_name": ai_data.get('scientific_name'),
                    "classification": ai_data.get('classification'),
                    "cultural": ai_data.get('cultural_methods'),
                    "biological": ai_data.get('biological_control'),
                    "sanitation": ai_data.get('sanitation'),
                    "mechanical": ai_data.get('mechanical_control'),
                    "chemical": ai_data.get('chemical_control'),
                    "pest_photo": url_for('static', filename=saved_image_db_path)
                })
            else:
                response_data['status_text'] = "AI Analysis Failed."

    except Exception as e:
        print(f"Error in api_status: {e}")
        response_data['status_text'] = "Error communicating with server."

    return jsonify(response_data)

# ================== STANDARD ROUTES ==================

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

        # Hardcoded Main Admin
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
    if 'admin' not in session:
        return redirect(url_for('login'))
    conn = None
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        c = conn.cursor()
        c.execute("SELECT * FROM pests")
        pests = c.fetchall()
        return render_template('admin_dashboard.html', pests=pests, admin_name=session['admin'])
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
    if 'admin' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        conn = None
        crop_type = request.form.get('crop_type', '').strip()
        common_name = request.form.get('common_name', '').strip()
        scientific_name = request.form.get('scientific_name', '').strip()
        order_name = request.form.get('order_name', '').strip()
        family = request.form.get('family', '').strip()
        classification = request.form.get('classification', '').strip()
        cultural_methods = request.form.get('cultural_methods', '').strip()
        biological_control = request.form.get('biological_control', '').strip()
        sanitation = request.form.get('sanitation', '').strip()
        mechanical_control = request.form.get('mechanical_control', '').strip()
        chemical_control = request.form.get('chemical_control', '').strip()
        image = request.files.get('image')

        if not crop_type or not common_name or not scientific_name or not classification or not cultural_methods or not chemical_control or not (image and image.filename):
            flash("Please ensure all required fields and an image are selected.", "danger")
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
                                             cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image)
                                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                          (crop_type, common_name, scientific_name, order_name, family, classification,
                                           cultural_methods, biological_control, sanitation, mechanical_control, chemical_control, image_filename_to_save))
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
    if 'admin' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 403

    pest_id = request.form.get('pest_id')
    image_file = request.files.get('image_file')

    if not pest_id or not image_file or not image_file.filename:
        return jsonify({'success': False, 'error': 'Missing Pest ID or image file.'}), 400

    try:
        filename_base, file_ext = os.path.splitext(image_file.filename)
        safe_filename = secure_filename(f"pest_{pest_id}{file_ext}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        image_file.save(filepath)
        image_db_path = os.path.join('uploads', safe_filename).replace('\\', '/')
        
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

        if not isinstance(pests_to_update, list):
            return jsonify({'success': False, 'error': "'pests' key not found or is not a list."}), 400
        
        if not pests_to_update:
            return jsonify({'success': True, 'message': 'No pest data to update.'})

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
                    pest['crop'], pest['common_name'], pest['scientific_name'], pest['order_name'],
                    pest['family'], pest['classification'], pest['cultural_methods'], pest['biological_control'],
                    pest['sanitation'], pest['mechanical_control'], pest['chemical_control'],
                    pest['id']
                ))
                success_count += 1
            except Exception as e:
                errors.append(f"Failed to update pest ID {pest.get('id', 'N/A')}: {str(e)}")

        conn.commit()
        
        if errors:
            return jsonify({
                'success': False, 
                'message': f"Completed with errors. {success_count} pests updated.",
                'errors': errors
            }), 500
        else:
            return jsonify({'success': True, 'message': 'All changes saved successfully!'})

    except Exception:
        return jsonify({'success': False, 'error': 'A critical server error occurred.'}), 500
    
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if 'admin' not in session:
        return redirect(url_for('login'))
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

        detected_pest_name = None
        pest_info = None

        try:
            filename = secure_filename(str(file.filename)) 
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('static', filename='uploads/' + filename)
            
            results = model(filepath)
            
            if results and len(results) > 0 and results[0].boxes and len(results[0].boxes.cls) > 0:
                best_conf_index = results[0].boxes.conf.argmax()
                class_index = int(results[0].boxes.cls[best_conf_index].item())
                detected_pest_name = results[0].names[class_index]
            
            if detected_pest_name:
                processed_name = detected_pest_name.strip().replace('-', ' ').replace('_', ' ').title()
                conn = get_db() 
                pest_info = conn.execute(
                    "SELECT * FROM pests WHERE common_name = ? COLLATE NOCASE",
                    (processed_name,)
                ).fetchone()

            if pest_info:
                return render_template('pest_upload.html', pest=pest_info, image_url=image_url)
            elif detected_pest_name:
                flash(f"Pest '{detected_pest_name.capitalize()}' detected, but no information found in the database.", "warning")
                return render_template('pest_upload.html', image_url=image_url)
            else:
                flash("No pest was detected in the uploaded image.", "info")
                return render_template('pest_upload.html', image_url=image_url)
                
        except Exception as e:
            flash(f"An unexpected error occurred during processing: {e}", "danger")
            return render_template('pest_upload.html', image_url=image_url)
            
    return render_template('pest_upload.html')

@app.route('/library')
def pest_library():
    conn = None 
    try:
        pest_db = os.path.join(DB_DIR, 'pests_add.db')
        conn = sqlite3.connect(pest_db)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM pests")
        pests = cur.fetchall()
        return render_template('pest_library.html', pests=pests)
    except Exception as e:
        flash("Could not load pest list.", "danger")
        return redirect(url_for('home'))
    finally:
        if conn: conn.close()

@app.route('/user')
def user_page():
    return render_template('user.html') 

@app.route('/index')
def index_page():
    return render_template('index.html')

if __name__ == '__main__':
    def release_cameras():
        global cam1, cam2, cam3
        if cam1.isOpened(): cam1.release()
        if cam2.isOpened(): cam2.release()
        if cam3.isOpened(): cam3.release()
        print("Cameras released.")
    
    atexit.register(release_cameras)
    app.run(debug=True)