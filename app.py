from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify, Response
import sqlite3, os, cv2
from ultralytics.models.yolo import YOLO 
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import atexit 
import re 
import datetime 
import time
import json 
import google.generativeai as genai 
from groq import Groq 
import PIL.Image  # <--- REQUIRED FOR VISION AI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_fallback_key')

# ================== AI CONFIGURATION ==================

# 1. GOOGLE GEMINI CONFIG
GENAI_API_KEY = os.getenv('GENAI_API_KEY')
if not GENAI_API_KEY:
    print("⚠️ WARNING: GENAI_API_KEY not found in .env file")
genai.configure(api_key=GENAI_API_KEY)

# 2. GROQ CONFIG
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def clean_json_text(text):
    """Helper to strip markdown formatting from AI responses."""
    text = text.strip()
    if text.startswith("```"):
        newline_index = text.find('\n')
        if newline_index != -1:
            text = text[newline_index+1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def fetch_pest_info_from_ai(pest_name, image_path=None):
    """
    1. If pest_name is 'Negative' AND image_path is provided: Uses Gemini Vision to Identify.
    2. Else: Uses Text-based lookup (Gemini -> Groq).
    """
    
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
    if pest_name.lower() == "negative" and image_path:
        print(f"👁️ AI Vision: Analyzing image to identify unknown organism...")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            # Open the saved image
            img = PIL.Image.open(image_path)
            
            vision_prompt = f"""
            Analyze this image. Identify the specific agricultural pest or insect present on the plant.
            If no pest is clearly visible, or if it is just a leaf/background, return a JSON with "common_name": "N/A".
            Otherwise, provide details in strict JSON format matching this structure: {json.dumps(json_structure)}.
            """
            
            response = model.generate_content([vision_prompt, img])
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

    print(f"🤖 AI Query: Asking Gemini about '{pest_name}'...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(system_prompt)
        cleaned_text = clean_json_text(response.text)
        return json.loads(cleaned_text)

    except Exception as e_gemini:
        print(f"❌ Gemini Text Failed: {e_gemini}")
        print("🔄 Switching to Groq fallback...")

        try:
            client = Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful agricultural AI assistant that outputs only valid JSON object."},
                    {"role": "user", "content": system_prompt}
                ],
                # UPDATED: Using the versatile model since 8b is decommissioned
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

# --- MODEL LOADING ---
model = YOLO('datapest.pt') 

# ================== DIAGNOSTIC CHECK ==================
try:
    print(f"\n✅ YOLO Model Loaded Successfully.")
    if model.names:
        print(f"   Detected classes: {model.names}")
except AttributeError:
    print("\n❌ CRITICAL ERROR: YOLO Model failed to load (datapest.pt not found or corrupted).")

# ================== LOGGING FUNCTION ==================

def log_detection_event(pest_name, image_path, detection_type):
    """Logs a successful pest detection event to the history table."""
    conn = get_db()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = session.get('admin', 'SYSTEM') 
    try:
        conn.execute("""
            INSERT INTO history (timestamp, yolo_name, image_path, user_session, detection_type)
            VALUES (?, ?, ?, ?, ?)
        """, (current_time, pest_name, image_path, user, detection_type))
        conn.commit()
        print(f"LOGGED: {pest_name} event saved to history.")
    except Exception as e:
        print(f"Error logging history: {e}")

# ================== DATABASE INIT ==================
def init_databases():
    """Initialize all databases and tables if they don't exist yet."""
    # === ADMIN DATABASE ===
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

    # === PEST DATABASE ===
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
    
    # === HISTORY DATABASE ===
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

# ================== CAMERA FEED WITH AI DETECTION ==================
camera = cv2.VideoCapture(0)

def generate_frames():
    global last_detected_pest
    global is_detection_running 
    global last_annotated_frame 
    global last_confidence 
    
    while True:
        success, frame = camera.read()
        if not success:
            frame = last_annotated_frame if last_annotated_frame is not None else frame
            if frame is None: break
        
        annotated_frame = frame
        found_pest = ""
        current_conf = 0.0 
        
        if is_detection_running:
            # Using 0.2 threshold to capture faint objects (Vision will filter bad ones)
            results = model(frame, stream=True, conf=0.2, verbose=False) 
            
            for r in results:
                annotated_frame = r.plot()
                last_annotated_frame = annotated_frame 
                
                if r.boxes and len(r.boxes.cls) > 0:
                    best_conf_index = r.boxes.conf.argmax()
                    class_index = int(r.boxes.cls[best_conf_index].item())
                    
                    found_pest = r.names[class_index]
                    current_conf = r.boxes.conf[best_conf_index].item()
                    
                    # Debug print
                    print(f"👀 Model sees: {found_pest} ({current_conf:.2f})")
                    break 
            
            last_detected_pest = found_pest
            last_confidence = current_conf 
        else:
            annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
            last_detected_pest = ""
            last_confidence = 0.0

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
@app.route('/video_feed')
def video_feed() -> Response:
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================== API CONTROL ROUTES ==================

@app.route('/api/start', methods=['POST'])
def api_start():
    global is_detection_running
    is_detection_running = True
    return jsonify({'status': 'Detection started'})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global is_detection_running
    is_detection_running = False
    global last_detected_pest
    global last_annotated_frame
    
    # Only save if it's NOT "Negative" (unless we have special handling, but usually stop means stop)
    if last_detected_pest and last_annotated_frame is not None and last_detected_pest.lower() != "negative":
        safe_name = secure_filename(f"{last_detected_pest}_{int(time.time())}.jpg")
        relative_path = os.path.join('history', safe_name)
        save_path = os.path.join(STATIC_FOLDER, relative_path)
        cv2.imwrite(save_path, last_annotated_frame)
        log_detection_event(last_detected_pest, relative_path, 'Live Feed')

    last_detected_pest = "" 
    last_annotated_frame = None 
    
    return jsonify({'status': 'Detection stopped'})

# ================== FETCH STATUS & PEST DETAILS API ==================

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
        "pest_name": "—", # Keep as dash to avoid stopping camera
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

    # === LOGIC: HANDLE "NEGATIVE" AS POTENTIAL UNKNOWN PEST ===
    if pest_name.lower() == "negative":
        # 1. Ignore low confidence noise
        if last_confidence <= 0.3:
             response_data['status_text'] = f"Scanning... (Background: {last_confidence:.2f})"
             return jsonify(response_data)
        
        # 2. High confidence "Negative" means something is there but YOLO doesn't know it.
        #    Trigger Vision AI to identify it.
        print(f"🚀 High Confidence Negative ({last_confidence:.2f}). Triggering Vision AI...")
        response_data['status_text'] = "Analyzing Unknown Object..."
        
        # Capture the image
        saved_image_db_path = ""
        saved_abs_path = ""
        if last_annotated_frame is not None:
            filename = secure_filename(f"unknown_{int(time.time())}.jpg")
            saved_abs_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(saved_abs_path, last_annotated_frame)
            saved_image_db_path = f"uploads/{filename}"

        # Call AI Vision
        ai_data = fetch_pest_info_from_ai("Negative", image_path=saved_abs_path)

        # 3. Handle Vision Result
        if ai_data and ai_data.get('common_name') and ai_data.get('common_name') not in ["N/A", "Negative"]:
            # Vision Successfully Identified it!
            new_name = ai_data.get('common_name')
            print(f"✅ Vision Identified: {new_name}")
            
            # Save to DB as the REAL name
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute('''INSERT INTO pests (
                    yolo_name, crop, common_name, scientific_name, order_name, family, 
                    classification, cultural_methods, biological_control, sanitation, 
                    mechanical_control, chemical_control, image
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    new_name, # Use the identified name
                    "Pineapple", 
                    new_name,
                    ai_data.get('scientific_name', 'N/A'),
                    ai_data.get('order_name', 'N/A'),
                    ai_data.get('family', 'N/A'),
                    ai_data.get('classification', 'N/A'),
                    ai_data.get('cultural_methods', 'N/A'),
                    ai_data.get('biological_control', 'N/A'),
                    ai_data.get('sanitation', 'N/A'),
                    ai_data.get('mechanical_control', 'N/A'),
                    ai_data.get('chemical_control', 'N/A'),
                    saved_image_db_path
                ))
                conn.commit()
            except Exception as e:
                print(f"DB Error: {e}")

            # Send data to frontend (This WILL trigger the stop because name is not empty)
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
            # AI saw nothing
            response_data['status_text'] = "Analysis Failed: No pest identified."
            response_data['pest_name'] = "—" # Keep running
            return jsonify(response_data)

    elif pest_name == "":
        response_data['status_text'] = "Scanning... (No pests detected)"
        return jsonify(response_data)

    # === LOGIC: KNOWN PESTS (YOLO DETECTED) ===
    
    pest_info = None
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Check DB
        cur.execute("SELECT * FROM pests WHERE yolo_name = ? COLLATE NOCASE LIMIT 1", (pest_name,))
        pest_info = cur.fetchone() 

        if pest_info:
            # === PEST FOUND IN DATABASE ===
            pest_dict = dict(pest_info)
            display_name = pest_dict.get('common_name', pest_name).strip() 
            image_path = pest_dict.get('image')
            
            response_data.update({
                "status_text": f"Pest Detected: {display_name}",
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
            # === KNOWN CLASS BUT NOT IN DB -> TEXT AI ===
            print(f"⚠️ Known class '{pest_name}' not in DB. Initiating Text AI...")
            
            # Save Image
            saved_image_db_path = ""
            if last_annotated_frame is not None:
                filename = secure_filename(f"auto_trained_{pest_name}_{int(time.time())}.jpg")
                abs_path = os.path.join(UPLOAD_FOLDER, filename)
                cv2.imwrite(abs_path, last_annotated_frame)
                saved_image_db_path = f"uploads/{filename}"

            # Call AI (Text only for known names)
            ai_data = fetch_pest_info_from_ai(pest_name)
            
            if ai_data:
                # Save to DB (omitted lengthy block for brevity, same as previous versions)
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

                response_data.update({
                    "status_text": f"AI Analyzed: {ai_data.get('common_name')}",
                    "pest_name": ai_data.get('common_name'),
                    # ... Map other fields ...
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


# ================== ORIGINAL ROUTES (Unchanged) ==================

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

        # --- Hardcoded MAIN ADMIN ---
        if username == 'Admin' and password == 'admin123':
            session['admin'] = username
            session['role'] = 'main'
            return redirect(url_for('admin_dashboard'))

        # --- Check in database ---
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
                return redirect(url_for('admin_dashboard'))
            else:
                return render_template('login.html', error="Invalid username or password.")
        except Exception:
            return render_template('login.html', error="A server error occurred during login.")
        finally:
            if conn:
                conn.close() 

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/admin_dashboard')
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
        if conn:
            conn.close()

@app.route('/add_pest', methods=['GET', 'POST'])
def add_pest():
    session.pop('_flashes', None)
    if 'admin' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        image_filename_to_save = None 
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

        # 1. Validation and Image Handling
        if not crop_type or not common_name or not scientific_name or not classification or not cultural_methods or not chemical_control or not (image and image.filename):
            flash("Please ensure all required fields and an image are selected.", "danger")
            return render_template('add_pest.html')

        try:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            image_filename_to_save = f'uploads/{filename}' 
            
            # 2. Save to DB 
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
            if conn:
                conn.close()
                
    return render_template('add_pest.html')

@app.route('/delete_pest/<int:pest_id>', methods=['POST'])
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
def upload_pest_image():
    """Handles the asynchronous upload of a pest image and updates the database."""
    if 'admin' not in session:
        return jsonify({'success': False, 'error': 'Authentication required'}), 403

    pest_id = request.form.get('pest_id')
    image_file = request.files.get('image_file')

    if not pest_id or not image_file or not image_file.filename:
        return jsonify({'success': False, 'error': 'Missing Pest ID or image file.'}), 400

    conn = None
    try:
        filename_base, file_ext = os.path.splitext(image_file.filename)
        safe_filename = secure_filename(f"pest_{pest_id}{file_ext}")
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        image_file.save(filepath)
        
        image_db_path = os.path.join('uploads', safe_filename).replace('\\', '/')
        
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("UPDATE pests SET image = ? WHERE id = ?", 
                     (image_db_path, pest_id))
        conn.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Image updated successfully!', 
            'image_url': url_for('static', filename=image_db_path)
        })

    except Exception as e:
        print(f"Image upload server error for ID {pest_id}: {e}")
        return jsonify({'success': False, 'error': f'Server error during image processing: {str(e)}'}), 500
    finally:
        pass

@app.route('/update_pests', methods=['POST'])
def update_pests():
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Invalid request format. Expected JSON.'}), 400

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
                required_keys = ['id', 'crop', 'common_name', 'scientific_name', 'order_name', 'family', 
                                 'classification', 'cultural_methods', 'biological_control', 'sanitation', 
                                 'mechanical_control', 'chemical_control']
                for key in required_keys:
                    if key not in pest:
                        raise KeyError(f"Missing required field in data: {key}")

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
                error_message = f"Failed to update pest ID {pest.get('id', 'N/A')}: {str(e)}"
                errors.append(error_message)

        conn.commit()
        
        if errors:
            return jsonify({
                'success': False, 
                'message': f"Completed with errors. {success_count} pests updated.",
                'errors': errors
            }), 500
        else:
            return jsonify({'success': True, 'message': 'All changes saved successfully!'})

    except Exception as e:
        return jsonify({'success': False, 'error': 'A critical server error occurred.'}), 500
    
@app.route('/register', methods=['GET', 'POST'])
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
            if conn:
                conn.close()

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
        print(f"Error fetching pest list: {e}")
        flash("Could not load pest list.", "danger")
        return redirect(url_for('home'))
    finally:
        if conn:
            conn.close()

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
                flash(f"Pest '{detected_pest_name.capitalize()}' detected, but no information found in the database. Please add details in the admin dashboard.", "warning")
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
        print(f"Error fetching pest list: {e}")
        flash("Could not load pest list.", "danger")
        return redirect(url_for('home'))
    finally:
        if conn:
            conn.close()

@app.route('/user')
def user_page():
    return render_template('user.html') 

@app.route('/index')
def index_page():
    return render_template('index.html')

if __name__ == '__main__':
    def release_camera():
        global camera
        if camera.isOpened():
            camera.release()
            print("Camera released.")
    
    atexit.register(release_camera)
    app.run(debug=True)