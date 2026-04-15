from ultralytics import YOLO

# 1. Load your local trained model
model = YOLO('native_updated.pt') 

# 2. Run validation on your test set
# 'split=test' ensures you are using the /test folder, not /valid
results = model.val(
    data='dataset/active_learning_data.yaml', 
    split='train',
    project='thesis_eval', 
    name='train_results'
)

# 3. Access numeric metrics directly
print(f"mAP 50: {results.box.map50}")
print(f"Precision: {results.box.mp}")
print(f"Recall: {results.box.mr}")

# 4. View the Confusion Matrix as a raw array (Optional)
print("Confusion Matrix Array:")
print(results.confusion_matrix.matrix)