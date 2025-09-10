import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ISLModel:
    def __init__(self, input_shape=(64, 64, 3), num_classes=35):  # 26 letters + 9 numbers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.label_encoder = LabelEncoder()
        
    def build_model(self):
        """
        Build a simpler, more stable CNN model for ISL recognition
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.0005):  # Reduced learning rate for stability
        """
        Compile the model with appropriate loss and optimizer
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Add gradient clipping
        )
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def preprocess_image(self, image_path, target_size=(64, 64)):
        """
        Preprocess image for model input
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        return img
    
    def load_dataset(self, data_dir):
        """
        Load and preprocess the entire dataset
        """
        images = []
        labels = []
        
        # Define class mapping
        class_mapping = {}
        class_idx = 0
        
        # Load numbers (1-9)
        for i in range(1, 10):
            class_mapping[str(i)] = class_idx
            class_idx += 1
        
        # Load letters (a-z)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            class_mapping[letter] = class_idx
            class_idx += 1
        
        print("Loading dataset...")
        
        # Load number images
        for num in range(1, 10):
            num_dir = os.path.join(data_dir, str(num))
            if os.path.exists(num_dir):
                for filename in os.listdir(num_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(num_dir, filename)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            images.append(img)
                            labels.append(class_mapping[str(num)])
        
        # Load letter images
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            letter_dir = os.path.join(data_dir, letter)
            if os.path.exists(letter_dir):
                for filename in os.listdir(letter_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(letter_dir, filename)
                        img = self.preprocess_image(img_path)
                        if img is not None:
                            images.append(img)
                            labels.append(class_mapping[letter])
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Dataset loaded: {len(X)} images, {len(np.unique(y))} classes")
        return X, y, class_mapping
    
    def train(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model on the ISL dataset
        """
        # Load dataset
        X, y, class_mapping = self.load_dataset(data_dir)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Compile model
        self.compile_model()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                'best_isl_model.h5', 
                save_best_only=True, 
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save class mapping
        self.class_mapping = class_mapping
        self.reverse_class_mapping = {v: k for k, v in class_mapping.items()}
        
        return history
    
    def predict(self, image_path):
        """
        Predict ISL sign from image
        """
        img = self.preprocess_image(image_path)
        if img is None:
            return None, 0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Get class label
        predicted_label = self.reverse_class_mapping.get(predicted_class, "Unknown")
        
        return predicted_label, confidence
    
    def predict_batch(self, image_paths):
        """
        Predict ISL signs from multiple images
        """
        predictions = []
        for img_path in image_paths:
            pred, conf = self.predict(img_path)
            predictions.append((pred, conf))
        return predictions
    
    def save_model(self, model_path='isl_model.h5'):
        """
        Save the trained model
        """
        self.model.save(model_path)
        # Save class mapping
        import pickle
        with open('class_mapping.pkl', 'wb') as f:
            pickle.dump(self.class_mapping, f)
    
    def load_model(self, model_path='isl_model.h5'):
        """
        Load a trained model
        """
        self.model = tf.keras.models.load_model(model_path)
        # Load class mapping
        import pickle
        with open('class_mapping.pkl', 'rb') as f:
            self.class_mapping = pickle.load(f)
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

if __name__ == "__main__":
    # Initialize model
    model = ISLModel()
    
    # Train model
    print("Training ISL model...")
    history = model.train('ISL_raw', epochs=50, batch_size=32)
    
    # Save model
    model.save_model()
    
    # Plot training history
    model.plot_training_history(history)
    
    print("Training completed!") 