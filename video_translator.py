import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
from collections import deque
import time

# --- Helper Functions (reused from test.py) ---
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input.
    Accepts an image array, not a path.
    """
    if image is None:
        return None
    img_bgr = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
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

    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        print(f"❌ Error: Model or mapping files not found.")
        print(f"   Please make sure '{model_path}' and '{mapping_path}' are in the current directory.")
        return None, None

    print(f"🤖 Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")

    print(f"📁 Loading class mapping from {mapping_path}...")
    with open(mapping_path, 'rb') as f:
        class_mapping = pickle.load(f)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    print(f"✅ Found {len(class_mapping)} classes.")
    
    return model, reverse_mapping

# --- Main Video Translation Function ---
def translate_video(video_path, model, reverse_mapping, temp_smoothing_window=10):
    """
    Translates an ISL video to text.
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    print(f"\n🎬 Starting video translation for {video_path}")
    print("💡 Press 'q' to exit the video window.")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default to 30 FPS if not found
    
    frame_interval = 1.0 / fps # Process every frame
    
    prediction_history = deque(maxlen=temp_smoothing_window)
    last_prediction_time = time.time()
    
    current_word = ""
    last_detected_char = ""
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame for model input
        preprocessed_frame = preprocess_image(frame)
        if preprocessed_frame is None:
            continue
            
        # Make a prediction
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        prediction = model.predict(preprocessed_frame, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        
        # Temporal Smoothing
        prediction_history.append(predicted_class_idx)
        
        # Find the most common prediction in the window
        most_common_pred = np.bincount(list(prediction_history)).argmax()
        
        # Check for change in prediction
        if reverse_mapping[most_common_pred] != last_detected_char:
            current_char = reverse_mapping[most_common_pred]
            
            # Simple word formation logic: if the character is different and
            # the confidence is high enough, add it to the word.
            if np.max(prediction) > 0.70: # Use a confidence threshold
                current_word += current_char
                last_detected_char = current_char
                print(f"Detected: {current_char}")
                print(f"Current Word: {current_word}")

        # Display output on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Predicted: {reverse_mapping[predicted_class_idx]}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Current Word: {current_word}', (50, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('ISL Translator', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Video translation completed!")

if __name__ == "__main__":
    MODEL_NAME = 'MobileNetV2'
    
    # Load the trained model and class mapping
    model, reverse_mapping = load_model_and_mapping(MODEL_NAME)
    
    if model and reverse_mapping:
        # Ask the user for the video file path
        video_file = input("Please enter the path to the video file (e.g., 'my_video.mp4'): ")
        translate_video(video_file, model, reverse_mapping)