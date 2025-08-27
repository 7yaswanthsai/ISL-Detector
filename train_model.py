import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50V2
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A

def create_data_augmentation():
    """
    Create data augmentation pipeline with more conservative settings
    """
    return A.Compose([
        # More conservative color augmentations
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        
        # Geometric augmentations
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        
        # Light noise (fixed parameter name)
        A.GaussNoise(var_limit=0.01, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_transfer_learning_model(base_model_name='MobileNetV2', input_shape=(224, 224, 3), num_classes=35, trainable_layers=50):
    """
    Create a transfer learning model using pre-trained weights
    
    Args:
        base_model_name: 'MobileNetV2', 'EfficientNetB0', or 'ResNet50V2'
        input_shape: Input image shape (224x224 recommended for pre-trained models)
        num_classes: Number of output classes
        trainable_layers: Number of top layers to make trainable for fine-tuning
    """
    print(f"🏗️ Creating transfer learning model with {base_model_name}...")
    
    # Create base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif base_model_name == 'ResNet50V2':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model: {base_model_name}")
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])
    
    print(f"✅ Base model created with {base_model.count_params():,} parameters")
    print(f"✅ Total model parameters: {model.count_params():,}")
    
    return model, base_model

def unfreeze_top_layers(model, base_model, num_layers=50):
    """
    Unfreeze top layers of the base model for fine-tuning
    """
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - num_layers

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"🔓 Unfrozen top {num_layers} layers for fine-tuning")
    print(f"   Trainable parameters: {model.count_params():,}")

def preprocess_with_skin_tone_handling(image_path, target_size=(224, 224)):
    """
    Preprocess image in grayscale (3-channel) to focus on hand shape/outline
    """
    # Read image
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    # Resize first to target to reduce memory
    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale and scale to [0,1]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype('float32') / 255.0

    # Convert to 3-channel grayscale expected by pretrained backbones
    img = np.stack([img_gray, img_gray, img_gray], axis=-1)

    return img

def normalize_skin_tone(img):
    """
    Normalize skin tone variations in the image with error handling
    """
    try:
        # Ensure image is not too large to avoid memory issues
        h, w = img.shape[:2]
        if h > 1024 or w > 1024:
            # Resize to manageable size first
            scale = min(1024/h, 1024/w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to LAB color space for better skin tone handling
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img
    except Exception as e:
        print(f"⚠️ Skin tone normalization failed, using original image: {e}")
        return img

def load_dataset_with_augmentation(data_dir, augment_factor=1, max_per_class=None):
    """
    Load dataset with better memory management and error handling
    """
    images = []
    labels = []
    failed_images = 0
    
    # Define class mapping
    class_mapping = {}
    class_idx = 0
    
    # Load numbers (1-9)
    for i in range(1, 10):
        class_mapping[str(i)] = class_idx
        class_idx += 1
    
    # Load letters (a-z or A-Z)
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        class_mapping[letter] = class_idx
        class_idx += 1
    
    print("📥 Loading dataset with improved memory management...")
    
    # Create augmentation pipeline (simplified to reduce memory usage)
    aug = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment_factor > 0 else None
    
    # Load number images
    for num in range(1, 10):
        num_dir = os.path.join(data_dir, str(num))
        if os.path.exists(num_dir):
            print(f"   Loading class {num}...")
            files = [f for f in os.listdir(num_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_per_class:
                files = files[:max_per_class]
                
            for filename in files:
                try:
                    img_path = os.path.join(num_dir, filename)
                    img = preprocess_with_skin_tone_handling(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_mapping[str(num)])
                        
                        # Add augmented versions (reduced to save memory)
                        if aug and augment_factor > 0:
                            try:
                                augmented = aug(image=img)['image']
                                images.append(augmented)
                                labels.append(class_mapping[str(num)])
                            except Exception as e:
                                print(f"⚠️ Augmentation failed for {filename}: {e}")
                    else:
                        failed_images += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {img_path}: {e}")
                    failed_images += 1
    
    # Load letter images (check both lowercase and uppercase)
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        # Try lowercase first
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.exists(letter_dir):
            # Try uppercase
            letter_dir = os.path.join(data_dir, letter.upper())
        
        if os.path.exists(letter_dir):
            print(f"   Loading class {letter}...")
            files = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_per_class:
                files = files[:max_per_class]
                
            for filename in files:
                try:
                    img_path = os.path.join(letter_dir, filename)
                    img = preprocess_with_skin_tone_handling(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(class_mapping[letter])
                        
                        # Add augmented versions (reduced to save memory)
                        if aug and augment_factor > 0:
                            try:
                                augmented = aug(image=img)['image']
                                images.append(augmented)
                                labels.append(class_mapping[letter])
                            except Exception as e:
                                print(f"⚠️ Augmentation failed for {filename}: {e}")
                    else:
                        failed_images += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {img_path}: {e}")
                    failed_images += 1
    
    if failed_images > 0:
        print(f"⚠️ {failed_images} images failed to load")
    
    # Convert to numpy arrays
    print("🔄 Converting to numpy arrays...")
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"✅ Dataset loaded: {len(X):,} images, {len(np.unique(y))} classes")
    print(f"   Memory usage: {X.nbytes / (1024**2):.1f} MB")
    
    return X, y, class_mapping

def train_transfer_learning_model(data_dir, base_model_name='MobileNetV2', epochs_phase1=20, epochs_phase2=30, batch_size=32, validation_split=0.2):
    """
    Train ISL model using transfer learning with two-phase training
    Phase 1: Train only the custom head with frozen base model
    Phase 2: Fine-tune with unfrozen top layers
    """
    print(f"🚀 Starting Transfer Learning Training with {base_model_name}")
    print("=" * 60)
    
    # Load dataset with augmentation (limited for memory efficiency)
    X, y, class_mapping = load_dataset_with_augmentation(data_dir, augment_factor=0, max_per_class=200)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    print(f"📊 Dataset Information:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Number of classes: {len(class_mapping)}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"✅ Class weights calculated for balanced training")
    
    # Create transfer learning model
    model, base_model = create_transfer_learning_model(
        base_model_name=base_model_name,
        input_shape=X_train.shape[1:],
        num_classes=len(class_mapping)
    )
    
    # Phase 1: Train with frozen base model
    print(f"\n🎯 PHASE 1: Training custom head (frozen base model)")
    print("-" * 50)
    
    # Compile for Phase 1 with higher learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Phase 1 callbacks
    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_isl_transfer_{base_model_name.lower()}_phase1.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Train Phase 1
    history_phase1 = model.fit(
        X_train, y_train,
        epochs=epochs_phase1,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase1,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print(f"✅ Phase 1 completed. Best validation accuracy: {max(history_phase1.history['val_accuracy']):.4f}")
    
    # Phase 2: Fine-tuning with unfrozen layers
    print(f"\n🎯 PHASE 2: Fine-tuning (unfrozen top layers)")
    print("-" * 50)
    
    # Unfreeze top layers
    unfreeze_top_layers(model, base_model, num_layers=50)
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Phase 2 callbacks
    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.3,
            patience=8,
            min_lr=1e-8,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_isl_transfer_{base_model_name.lower()}_final.h5',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Train Phase 2
    history_phase2 = model.fit(
        X_train, y_train,
        epochs=epochs_phase2,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_phase2,
        class_weight=class_weight_dict,
        verbose=1,
        initial_epoch=len(history_phase1.history['loss'])
    )
    
    print(f"✅ Phase 2 completed. Best validation accuracy: {max(history_phase2.history['val_accuracy']):.4f}")
    
    # Combine histories
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    # Create a simple object to hold the combined history
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history_obj = CombinedHistory(combined_history)
    
    # Save class mapping and model info
    model.class_mapping = class_mapping
    model.reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    return model, combined_history_obj, X_val, y_val, base_model_name

def evaluate_model_performance(model, X_val, y_val):
    """
    Evaluate model performance and create detailed metrics
    """
    # Predict on validation set
    predictions = model.model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_val)
    
    # Calculate per-class accuracy
    unique_classes = np.unique(y_val)
    class_accuracy = {}
    
    for class_idx in unique_classes:
        class_mask = y_val == class_idx
        class_acc = np.mean(predicted_classes[class_mask] == y_val[class_mask])
        class_label = model.reverse_class_mapping.get(class_idx, f"Class_{class_idx}")
        class_accuracy[class_label] = class_acc
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-class Accuracy:")
    for class_name, acc in class_accuracy.items():
        print(f"{class_name}: {acc:.4f}")
    
    return accuracy, class_accuracy

def plot_training_results(history, class_accuracy):
    """
    Plot comprehensive training results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Per-class accuracy
    classes = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())
    
    ax3.bar(range(len(classes)), accuracies)
    ax3.set_title('Per-Class Accuracy')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    
    # Accuracy distribution
    ax4.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Accuracy Distribution')
    ax4.set_xlabel('Accuracy')
    ax4.set_ylabel('Number of Classes')
    
    plt.tight_layout()
    plt.savefig('training_results_skin_tone.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🤟 ISL Transfer Learning Training")
    print("=" * 50)
    
    # Choose your model: 'MobileNetV2', 'EfficientNetB0', or 'ResNet50V2'
    # MobileNetV2: Fastest, good for mobile deployment
    # EfficientNetB0: Best accuracy-to-size ratio
    # ResNet50V2: Most accurate but larger
    
    BASE_MODEL = 'MobileNetV2'  # Change this to try different models
    
    print(f"🎯 Selected base model: {BASE_MODEL}")
    print(f"📁 Dataset directory: ISL_raw")
    
    # Train model with transfer learning
    model, history, X_val, y_val, model_name = train_transfer_learning_model(
        data_dir='ISL_raw',
        base_model_name=BASE_MODEL,
        epochs_phase1=15,  # Phase 1: Train custom head
        epochs_phase2=25,  # Phase 2: Fine-tune
        batch_size=32
    )
    
    # Save the final model
    final_model_path = f'isl_transfer_{model_name.lower()}_final.keras'
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Save class mapping
    import pickle
    # Create class mapping from dataset structure
    classes = sorted([d for d in os.listdir('ISL_raw') if os.path.isdir(os.path.join('ISL_raw', d))])
    class_mapping = {class_name: idx for idx, class_name in enumerate(classes)}
    
    with open(f'class_mapping_transfer_{model_name.lower()}.pkl', 'wb') as f:
        pickle.dump(class_mapping, f)
    print(f"💾 Class mapping saved: class_mapping_transfer_{model_name.lower()}.pkl")
    
    # Evaluate performance
    if X_val is not None and y_val is not None:
        accuracy, class_accuracy = evaluate_model_performance(model, X_val, y_val)
        plot_training_results(history, class_accuracy)
    
    # Final summary
    best_val_acc = max(history.history['val_accuracy'])
    total_epochs = len(history.history['loss'])
    
    print(f"\n🎉 Transfer Learning Training Completed!")
    print(f"=" * 50)
    print(f"📊 Final Results:")
    print(f"   Base Model: {BASE_MODEL}")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Total Training Epochs: {total_epochs}")
    print(f"   Final Model: {final_model_path}")
    
    if best_val_acc > 0.90:
        print(f"\n🏆 Outstanding results! Your model is production-ready!")
    elif best_val_acc > 0.80:
        print(f"\n✅ Excellent results! Great improvement over from-scratch training!")
    elif best_val_acc > 0.60:
        print(f"\n👍 Good results! Much better than your previous 28-30% accuracy!")
    else:
        print(f"\n🤔 Results need improvement. Try a different base model or more data.")
    
    print(f"\n💡 To use your trained model:")
    print(f"   model = tf.keras.models.load_model('{final_model_path}')")
    print(f"   # Load class mapping and make predictions!")
    
    print(f"\n🔄 Want to try a different model? Change BASE_MODEL to:")
    print(f"   - 'MobileNetV2' (fastest, mobile-friendly)")
    print(f"   - 'EfficientNetB0' (best balance)")
    print(f"   - 'ResNet50V2' (most accurate)") 