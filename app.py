from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify, Response
import sqlite3, os, cv2, threading, atexit, re, datetime, time, json
import numpy as np 
from ultralytics.models.yolo import YOLO 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
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
app.permanent_session_lifetime = timedelta(seconds=int(os.getenv('SESSION_TIMEOUT_SEC', '900')))

# ================== AI CONFIGURATION ==================
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
if not GENAI_API_KEY:
    print("⚠️ WARNING: GENAI_API_KEY not found in .env file")

try:
    gemini_client = genai.Client(api_key=GENAI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    gemini_client = None

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# ================== GLOBAL STATE & LOCKS ==================
# Thread locks are essential for preventing database corruption and race conditions
frame_lock = threading.Lock()
db_lock = threading.Lock()

last_detected_pest = ""           
is_detection_running = False      
last_annotated_frame = None       
last_confidence = 0.0
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

# ================== HELPERS ==================

def clean_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        newline_index = text.find('\n')
        if newline_index != -1:
            text = text[newline_index+1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def get_db():
    if 'db' not in g:
        # Timeout helps avoid "database is locked" errors
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
        is_api = request.path.startswith('/api') or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', '')
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

# --- AI FETCHING ---
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

    # SCENARIO A: VISION IDENTIFICATION (Unknown/Negative)
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
                model='gemini-1.5-flash-001',
                contents=[vision_prompt, img]
            )
            cleaned_text = clean_json_text(response.text)
            return json.loads(cleaned_text)
        except Exception as e_vision:
            print(f"❌ Vision Analysis Failed: {e_vision}")
            return None

    # SCENARIO B: TEXT LOOKUP (Known Name)
    system_prompt = f"""
    You are an agricultural expert. I have detected a pest named '{pest_name}' on a Pineapple crop.
    Provide details in strict JSON format matching this structure: {json.dumps(json_structure)}.
    Do not add conversational text. If data is unknown, use "N/A".
    """

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash-001',
                contents=system_prompt
            )
            cleaned_text = clean_json_text(response.text)
            return json.loads(cleaned_text)
        except Exception as e_gemini:
             print(f"❌ Gemini Text Failed: {e_gemini}")

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

# --- DB INIT ---
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
    """Checks if yolo_name column exists and adds it if missing."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Get list of current columns
        cursor.execute("PRAGMA table_info(pests)")
        columns = [info[1] for info in cursor.fetchall()]
        
        # If 'yolo_name' is missing, add it
        if 'yolo_name' not in columns:
            print("🔧 MAINTENANCE: 'yolo_name' column missing. Adding it now...")
            cursor.execute("ALTER TABLE pests ADD COLUMN yolo_name TEXT")
            conn.commit()
            print("✅ Database patched! 'yolo_name' column added successfully.")
        else:
            print("✅ Database schema is up to date.")
            
        conn.close()
    except Exception as e:
        print(f"❌ Error patching database: {e}")

patch_database_schema()

# --- MODEL ---
try:
    model = YOLO('datapest.pt') 
except:
    print("⚠️ WARNING: datapest.pt not found. Detection will fail.")
    model = None

def log_detection_event(pest_name, image_path, detection_type):
    # Use separate connection with lock for background threads
    with db_lock:
        try:
            conn = sqlite3.connect(DATABASE, timeout=10)
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                user = session.get('admin', 'SYSTEM') 
            except:
                user = 'SYSTEM'
            
            # Fix path separators for Windows
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
    global last_logged_pest
    global last_annotated_frame
    
    if pest_name and pest_name != last_logged_pest:
        frame_copy = None
        with frame_lock:
            if last_annotated_frame is not None:
                frame_copy = last_annotated_frame.copy()
        
        if frame_copy is not None:
            try:
                safe_name = secure_filename(f"{pest_name}_{int(time.time())}.jpg")
                relative_path = os.path.join('history', safe_name)
                save_path = os.path.join(STATIC_FOLDER, 'history', safe_name)
                cv2.imwrite(save_path, frame_copy)
                
                db_path = f"history/{safe_name}"
                log_detection_event(pest_name, db_path, 'Continuous Feed')
                
                last_logged_pest = pest_name 
                print(f"✅ Auto-logged: {pest_name}")
            except Exception as e:
                print(f"Error continuous logging: {e}")

# ================== CAMERA LOGIC (LAZY LOADING & ANTI-SPAM) ==================

# Lazy load cameras to prevent init crashes
cams = {0: None, 1: None, 2: None}

def get_camera(index):
    global cams
    if cams[index] is None or not cams[index].isOpened():
        cams[index] = cv2.VideoCapture(index, cv2.CAP_DSHOW) # CAP_DSHOW improves Windows compatibility
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
        
        if is_detection_running and model:
            # Lower confidence to ensure we catch the beetle
            results = model(frame, stream=True, conf=0.25, verbose=False) 
            
            # Variables to find the best pest in THIS frame
            best_conf = 0.0
            best_pest = None

            for r in results:
                annotated_frame = r.plot()
                with frame_lock:
                    last_annotated_frame = annotated_frame 

                # Loop through every box found in the frame
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = r.names[cls_id]

                    # --- CRITICAL FIX ---
                    # 1. Ignore "Negative" completely
                    # 2. Ignore "Unknown" if you want to force DB lookups
                    if label.lower() in ["negative", "unknown"]:
                        continue
                    
                    # 3. Pick the pest with the highest confidence
                    if conf > best_conf:
                        best_conf = conf
                        best_pest = label
            
            # Only update the global variable if we found a REAL pest (not Negative)
            if best_pest:
                last_detected_pest = best_pest
                last_confidence = best_conf
                
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
            # FIX: Wait 10 seconds to stop terminal spam
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
            # FIX: Wait 10 seconds to stop terminal spam
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

# ================== API ROUTES (OPEN ACCESS) ==================

@app.route('/api/start', methods=['POST'])
# @login_required  <-- REMOVED
def api_start():
    global is_detection_running, last_logged_pest
    is_detection_running = True
    last_logged_pest = None 
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
# @login_required  <-- REMOVED
def api_stop():
    global is_detection_running, last_detected_pest
    is_detection_running = False
    last_detected_pest = "" 
    return jsonify({'status': 'Detection stopped'})

@app.route('/api/status')
# @login_required
def api_status():
    global last_detected_pest, is_detection_running, last_annotated_frame, last_confidence
    
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

    # BLOCK NEGATIVE / UNKNOWN
    if pest_name.lower() in ['negative', 'unknown']:
         response_data['status_text'] = "Scanning... (Filtering noise)"
         return jsonify(response_data)

    try:
        conn = get_db()
        cur = conn.cursor()
        pest_info = None
        
        # 1. Try Exact YOLO Name (e.g., "african-snail")
        try:
            cur.execute("SELECT * FROM pests WHERE yolo_name = ? COLLATE NOCASE LIMIT 1", (pest_name,))
            pest_info = cur.fetchone()
        except:
            pass # Column might be missing if patch didn't run yet

        # 2. Try Title Case Match (e.g., "African Snail")
        if not pest_info:
            formatted_name = pest_name.replace('-', ' ').replace('_', ' ').title()
            print(f"🔄 Searching Exact: '{formatted_name}'")
            cur.execute("SELECT * FROM pests WHERE common_name LIKE ? LIMIT 1", (formatted_name,))
            pest_info = cur.fetchone()

        # 3. Try Partial Match (e.g., "African Snail" will find "Giant African Snail")
        if not pest_info:
            print(f"🔎 Searching Partial: '%{formatted_name}%'")
            cur.execute("SELECT * FROM pests WHERE common_name LIKE ? LIMIT 1", (f"%{formatted_name}%",))
            pest_info = cur.fetchone()

        if pest_info:
            print(f"✅ FOUND: {pest_info['common_name']}")
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
        else:
             print(f"❌ '{pest_name}' NOT found in DB")
            
    except Exception as e:
        print(f"❌ Database Error: {e}")

    response_data['status_text'] = f"Detected '{pest_name}' (Not in DB)"
    response_data['pest_name'] = pest_name
    return jsonify(response_data)

# ================== STANDARD WEB ROUTES ==================

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

        # Hardcoded Fallback
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

        if not crop_type or not common_name or not (image and image.filename):
            flash("Please ensure required fields and an image are selected.", "danger")
            return render_template('add_pest.html')

        try:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            # FIX: Ensure forward slashes for DB path
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

        if not isinstance(pests_to_update, list):
            return jsonify({'success': False, 'error': "'pests' key not found or is not a list."}), 400
        
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
            return jsonify({
                'success': False, 
                'message': f"Completed with errors. {success_count} updated.",
                'errors': errors
            })
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

        detected_pest_name = None
        pest_info = None

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
        global cams
        for i in cams:
            if cams[i] and cams[i].isOpened():
                cams[i].release()
        print("Cameras released.")
    
    atexit.register(release_cameras)
    # Threaded=True is essential for simultaneous camera streaming + API calls
    app.run(debug=True, use_reloader=False, threaded=True)