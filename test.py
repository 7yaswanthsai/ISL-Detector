import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import pickle

# Centralized preprocessing function for consistency
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input, including skin tone handling and resizing.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = np.stack([img_gray, img_gray, img_gray], axis=-1)
    img = img.astype('float32') / 255.0
    return img

def load_model_and_mapping(model_name):
    """
    Load the trained model and its corresponding class mapping.
    """
    model_path = f'isl_transfer_{model_name.lower()}_final.keras'
    mapping_path = f'class_mapping_transfer_{model_name.lower()}.pkl'

    print(f"🤖 Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")

    print(f"📁 Loading class mapping from {mapping_path}...")
    with open(mapping_path, 'rb') as f:
        class_mapping = pickle.load(f)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    print(f"✅ Found {len(class_mapping)} classes: {list(class_mapping.keys())}")
    
    return model, reverse_mapping

def predict_image(model, reverse_mapping, image_path):
    """
    Predict the ISL sign from a single image.
    """
    img = preprocess_image(image_path)
    if img is None:
        return None, 0, None
    
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    
    predicted_class = int(np.argmax(pred))
    predicted_label = reverse_mapping[predicted_class]
    confidence = float(np.max(pred))
    
    top_3_indices = np.argsort(pred[0])[-3:][::-1]
    top_3 = [(reverse_mapping[i], float(pred[0][i])) for i in top_3_indices]
    
    return predicted_label, confidence, top_3

def test_single_image(model, reverse_mapping):
    """
    Test a single image.
    """
    test_image_path = 'c1.png'
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("💡 Please place your test image in the current directory or update the path")
        return
    
    print(f"🔍 Testing image: {test_image_path}")
    print("=" * 60)
    predicted_label, confidence, top_3 = predict_image(model, reverse_mapping, test_image_path)
    
    if predicted_label:
        print(f"   Predicted: {predicted_label}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Top 3: {top_3[0][0]} ({top_3[0][1]:.2%})")

def get_ground_truth_label(filename):
    """
    Extracts the ground truth label from a filename.
    Assumes filenames are in the format 'X1.png' or 'Y (1).jpg'
    """
    # Remove file extension and any numbering
    name = os.path.splitext(filename)[0]
    name = name.split(' ')[0] # Handles 'D (1)' format
    # The first character is typically the label
    label = name[0]
    return label.lower()

def test_with_ground_truth(model, reverse_mapping):
    """
    Tests all images in the directory and provides accuracy report.
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(ext))
        
    if not test_images:
        print("❌ No test images found in current directory")
        print("💡 Supported formats: .jpg, .jpeg, .png, .bmp")
        return

    correct_predictions = 0
    total_images = len(test_images)
    
    print(f"🔍 Found {total_images} test images")
    print("=" * 60)

    for i, image_path in enumerate(test_images, 1):
        predicted_label, confidence, _ = predict_image(model, reverse_mapping, image_path)
        ground_truth = get_ground_truth_label(os.path.basename(image_path))

        is_correct = predicted_label == ground_truth
        if is_correct:
            correct_predictions += 1
            status = "✅ Correct"
        else:
            status = "❌ Incorrect"
        
        print(f"{i:02d}. {os.path.basename(image_path)} -> Predicted: {predicted_label} ({confidence:.2%}) | Actual: {ground_truth} | {status}")
    
    accuracy = (correct_predictions / total_images) * 100
    print("\n" + "=" * 60)
    print(f"📊 Final Accuracy Report:")
    print(f"   Correct Predictions: {correct_predictions} / {total_images}")
    print(f"   Overall Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
if __name__ == "__main__":
    MODEL_NAME = 'MobileNetV2'
    
    print("🤟 ISL Model Testing")
    print("=" * 60)
    print("💡 Using the same preprocessing as training.")
    print("=" * 60)
    
    model, reverse_mapping = load_model_and_mapping(MODEL_NAME)
    
    test_single_image(model, reverse_mapping)
    print("\n" + "=" * 60)
    response = input("\n🔍 Do you want to test all images in current directory? (y/n): ").lower()
    if response in ['y', 'yes']:
        test_with_ground_truth(model, reverse_mapping)
    print("\n✅ Testing completed!")