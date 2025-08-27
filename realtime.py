import tensorflow as tf
import numpy as np
import cv2
import os
import pickle
from collections import deque
from googletrans import Translator, LANGUAGES
import time

# --- Helper Functions (reused from previous scripts) ---
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
        return None, None, None

    print(f"🤖 Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")

    print(f"📁 Loading class mapping from {mapping_path}...")
    with open(mapping_path, 'rb') as f:
        class_mapping = pickle.load(f)
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    print(f"✅ Found {len(class_mapping)} classes.")
    
    return model, reverse_mapping, class_mapping

# --- Main Real-Time Translation Function ---
def real_time_translator(model, reverse_mapping, class_mapping):
    """
    Performs real-time ISL translation from webcam feed.
    """
    print("\nStarting real-time ISL translation...")
    print("💡 Press 'q' to exit the window.")
    
    cap = cv2.VideoCapture(0) # Use 0 for the default webcam
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    translator = Translator()

    # Define supported Indian languages for translation
    indian_languages = {
        'Hindi': 'hi',
        'Bengali': 'bn',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Marathi': 'mr',
        'Gujarati': 'gu',
        'Punjabi': 'pa',
        'Urdu': 'ur',
        'Kannada': 'kn'
    }
    
    prediction_history = deque(maxlen=20)
    current_word = []
    last_detected_char = ""
    
    # Initialize translation variables
    translated_words = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)

        preprocessed_frame = preprocess_image(frame)
        
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        prediction = model.predict(preprocessed_frame, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        
        prediction_history.append(predicted_class_idx)
        most_common_pred = np.bincount(list(prediction_history)).argmax()

        # Check for a new stable sign
        if np.max(prediction) > 0.70 and reverse_mapping[most_common_pred] != last_detected_char:
            char_to_add = reverse_mapping[most_common_pred]
            
            # Simple word formation: 'space' sign or a pause
            if char_to_add == ' ': # Assuming a space sign exists
                if current_word:
                    text_to_translate = "".join(current_word)
                    print(f"Translating word: {text_to_translate}")
                    
                    translated_words = {}
                    for lang_name, lang_code in indian_languages.items():
                        try:
                            translated_text = translator.translate(text_to_translate, dest=lang_code).text
                            translated_words[lang_name] = translated_text
                        except Exception as e:
                            print(f"⚠️ Translation to {lang_name} failed: {e}")
                            translated_words[lang_name] = "N/A"
                    
                    print(f"Translations: {translated_words}")
                    current_word = []
            else:
                current_word.append(char_to_add)
            
            last_detected_char = char_to_add
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, f'Sign: {reverse_mapping[predicted_class_idx]}', (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.putText(frame, f'Word: {"".join(current_word)}', (50, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        y_pos = 150
        for lang_name, translated_text in translated_words.items():
            cv2.putText(frame, f'{lang_name}: {translated_text}', (50, y_pos), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 30
            
        cv2.imshow('ISL Translator - Real-Time', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Real-time translation stopped.")

if __name__ == "__main__":
    MODEL_NAME = 'MobileNetV2'
    
    model, reverse_mapping, class_mapping = load_model_and_mapping(MODEL_NAME)
    
    if model and reverse_mapping:
        if ' ' not in class_mapping:
            print("⚠️ Warning: No 'space' sign found in your class mapping. Word formation may be difficult.")
            print("   Word formation will rely on consecutive, different signs.")
        
        real_time_translator(model, reverse_mapping, class_mapping)