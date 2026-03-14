import os
import json
import shutil
import requests
from ultralytics import YOLO

FLASK_URL = "http://127.0.0.1:5000"

def run_active_learning():
    dataset_dir = os.path.abspath('dataset')
    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels')
    yaml_path = os.path.join(dataset_dir, 'active_learning.yaml')
    classes_file = os.path.join(dataset_dir, 'classes.json')

    # 1. Check if we have new data to train on
    if not os.path.exists(classes_file) or not os.listdir(img_dir):
        print("No new dataset found. Skipping training.")
        return

    with open(classes_file, 'r') as f:
        new_classes = json.load(f)

    # 2. Fix dynamic label IDs for YOLO (Mapping 100+ back to 0, 1, 2...)
    id_mapping = {}
    class_names = []
    for idx, (name, old_id) in enumerate(sorted(new_classes.items(), key=lambda item: item[1])):
        id_mapping[str(old_id)] = str(idx)
        class_names.append(name)

    for txt_name in os.listdir(lbl_dir):
        if txt_name.endswith('.txt'):
            filepath = os.path.join(lbl_dir, txt_name)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            with open(filepath, 'w') as f: # Overwrite with corrected 0-indexed IDs
                for line in lines:
                    parts = line.strip().split()
                    if parts and parts[0] in id_mapping:
                        parts[0] = id_mapping[parts[0]]
                        f.write(" ".join(parts) + "\n")

    # 3. Build the YOLO YAML configuration
    yaml_content = f"""
train: {img_dir}
val: {img_dir}

nc: {len(class_names)}
names: {class_names}
    """
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("==================================================")
    print("🚦 PAUSING LIVE INFERENCE TO FREE UP CPU")
    print("==================================================")
    try:
        requests.post(f"{FLASK_URL}/api/maintenance/pause", timeout=5)
    except requests.exceptions.RequestException:
        print("⚠️ Could not contact Flask server. Is it running?")
        return

    # 4. Train the model with strict Mini PC constraints
    model = YOLO('native.pt')
    
    # We use very strict parameters to prevent the Dell Wyse from crashing
    results = model.train(
        data=yaml_path,
        epochs=15,          
        imgsz=320,          
        device='cpu',       
        workers=1,          
        batch=2,            
        project='runs',
        name='active_update',
        exist_ok=True       
    )

    # 5. Move the newly trained brain to the root folder
    new_weights = os.path.join('runs', 'active_update', 'weights', 'best.pt')
    if os.path.exists(new_weights):
        shutil.copy(new_weights, 'new_best.pt')
        print("✅ Training complete! 'new_best.pt' generated.")

    print("==================================================")
    print("▶️ RESUMING LIVE INFERENCE WITH NEW MODEL")
    print("==================================================")
    try:
        requests.post(f"{FLASK_URL}/api/maintenance/resume", json={'model_path': 'new_best.pt'}, timeout=10)
    except requests.exceptions.RequestException:
        print("⚠️ Could not contact Flask server to resume.")

if __name__ == '__main__':
    run_active_learning()