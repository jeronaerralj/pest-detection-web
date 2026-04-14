# train_update.py
import os
import json
import yaml
import shutil
import requests
from ultralytics import YOLO

# (If native.pt class 0 was "Mealybug", Mealybug must be first here).
BASE_CLASSES = [
    "Cutworm Larva",
    "Cutworm Moth",
    "Flower Thrips",
    "Gray Borer",
    "Gray Borer Generic",
    "Mealybug",
    "Mealybug Cluster",
    "Oriental Fruit Fly",
    "Rhinoceros Beetle",
    "Slug Caterpillar",
    "Weaver Ant",
    "Weaver Ant Cluster"
]

DATASET_DIR = os.path.abspath('dataset')
CLASSES_JSON_PATH = os.path.join(DATASET_DIR, 'classes.json')
YAML_OUTPUT_PATH = os.path.join(DATASET_DIR, 'active_learning_data.yaml')

def generate_yaml():
    print("📝 Generating dynamic data.yaml...")

    # Start with the base classes
    all_classes = list(BASE_CLASSES)
    
    # Read dynamically added classes from the Flask app
    if os.path.exists(CLASSES_JSON_PATH):
        with open(CLASSES_JSON_PATH, 'r') as f:
            new_classes = json.load(f)

            # Sort new classes from JSON to keep mapping stable
            sorted_new = sorted(new_classes.items(), key=lambda item: item[1])
            for name, old_id in sorted_new:
                        if name not in all_classes:
                            all_classes.append(name)     
                
    # Build the YAML structure
    yaml_data = {
        'path': DATASET_DIR,
        'train': 'images',  # YOLO will look in dataset/images
        'val': 'images',    # For fine-tuning, validating on the training set is acceptable
        'nc': len(all_classes),
        'names': all_classes
    }
    
    # Write to file
    with open(YAML_OUTPUT_PATH, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"✅ YAML generated successfully with {len(all_classes)} total classes.")
    return YAML_OUTPUT_PATH

def start_training():
    print("🚀 Initiating Background Training Sequence...")
    
    # 1. Generate the config
    yaml_path = generate_yaml()

    updated_model_path = 'native_updated.pt'
    base_model_path = 'native.pt'

    # If we already have an updated model, use that as the base for the next round
    if os.path.exists(updated_model_path):
        print(f"🧠 Loading the PREVIOUSLY UPDATED model ({updated_model_path})...")
        model = YOLO(updated_model_path)
    else:
        print(f"🧠 Loading the ORIGINAL base model ({base_model_path})...")
        model = YOLO(base_model_path)
    
    print("⚙️ Training YOLO model...")
    results = model.train(
        data=yaml_path,
        epochs=30,          # Fewer epochs needed since we are just fine-tuning
        imgsz=640,
        batch=8,
        project='runs/active_learning',
        name='update',
        exist_ok=True,
        freeze=10           # Freezes the backbone to prevent catastrophic forgetting
    )
    
    # 4. Safely copy the newly trained weights
    new_weights = os.path.join('runs', 'detect', 'runs', 'active_learning', 'update', 'weights', 'best.pt')
    updated_model_path = 'native_updated.pt'
    
    if os.path.exists(new_weights):
        shutil.copy(new_weights, updated_model_path)
        print(f"🎉 Training complete! New model saved as {updated_model_path}")
        
        # 5. Ping Flask to hot-swap the model
        try:
            response = requests.post('http://127.0.0.1:5000/api/maintenance/resume', 
                                     json={'model_path': updated_model_path})
            if response.status_code == 200:
                print("✅ Flask system successfully hot-swapped to the new model.")
            else:
                print(f"⚠️ Flask returned status {response.status_code} on resume.")
        except Exception as e:
            print(f"⚠️ Failed to auto-resume Flask (Is the server running?): {e}")

if __name__ == "__main__":
    start_training()