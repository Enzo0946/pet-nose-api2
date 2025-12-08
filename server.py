# server.py - TensorFlow Lite Version (Railway Compatible) - WITH JSON SERIALIZATION FIX
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, storage
import json
import tempfile
import traceback
import tensorflow as tf
from typing import List, Dict, Any, Union
import urllib.request
import random

print("=" * 70)
print("🚀 Starting PawTag Backend - TensorFlow Lite Version")
print("=" * 70)

# ---------------------------
# MEMORY OPTIMIZATION FOR RAILWAY
# ---------------------------
print("\n💾 Memory optimization for Railway...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for compatibility

# Force TensorFlow to use CPU only (Railway doesn't have GPU)
tf.config.set_visible_devices([], 'GPU')
# Reduce TensorFlow threads to save memory
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("   ✅ Memory optimization applied")

# Configuration
DEFAULT_THRESHOLD = 0.6
DEFAULT_MOBILENET_WEIGHT = 0.8
DEFAULT_ORB_WEIGHT = 0.1
DEFAULT_FACE_WEIGHT = 0.1

print(f"\n⚖️ Feature weights:")
print(f"   MobileNet (TensorFlow): {DEFAULT_MOBILENET_WEIGHT}")
print(f"   ORB: {DEFAULT_ORB_WEIGHT}")
print(f"   Facial: {DEFAULT_FACE_WEIGHT}")

# ---------------------------
# IMPROVED REJECTION THRESHOLDS
# ---------------------------
print("\n🛡️ Setting up IMPROVED pet rejection thresholds...")

# Primary threshold - Most important for rejection
MIN_MATCH_SCORE_FOR_DB = 0.60  # Increased from 0.55

# Secondary check - Only reject if score is very low
MIN_ACCEPTABLE_SCORE = 0.35  # Lowered from 0.40

# Score difference - More lenient for similar pets
UNCERTAIN_SCORE_DIFF = 0.08  # Lowered from 0.15 (much more lenient)

# NEW: Absolute minimum score to even consider rejection
ABSOLUTE_MIN_SCORE = 0.25

# NEW: High confidence threshold - if above this, always accept
HIGH_CONFIDENCE_THRESHOLD = 0.75

print(f"   Database acceptance threshold: {MIN_MATCH_SCORE_FOR_DB}")
print(f"   High confidence threshold: {HIGH_CONFIDENCE_THRESHOLD}")
print(f"   Minimum acceptable score: {MIN_ACCEPTABLE_SCORE}")
print(f"   Absolute minimum score: {ABSOLUTE_MIN_SCORE}")
print(f"   Uncertain match threshold: {UNCERTAIN_SCORE_DIFF} (more lenient)")

# ---------------------------
# UTILITY FUNCTIONS FOR JSON SERIALIZATION
# ---------------------------
def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ---------------------------
# LIGHT AUGMENTATION FUNCTIONS (Memory Efficient)
# ---------------------------
print("\n🌀 Initializing Light Augmentation System...")

def apply_light_augmentation(color_img, gray_img):
    """
    Apply memory-efficient augmentations
    Returns: List of (color_augmented, gray_augmented, augmentation_name)
    """
    augmentations = []
    
    # Always include original
    augmentations.append((color_img.copy(), gray_img.copy(), "original"))
    
    # 1. Horizontal Flip (most effective for pets, doubles data)
    color_flip = cv2.flip(color_img, 1)
    gray_flip = cv2.flip(gray_img, 1)
    augmentations.append((color_flip, gray_flip, "flip"))
    
    # 2. Mild Brightness (only 50% chance to save memory)
    if random.random() > 0.5:  # 50% probability
        # Small brightness adjustment (-10% to +10%)
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-10, 10)
        color_bright = cv2.convertScaleAbs(color_img, alpha=alpha, beta=beta)
        augmentations.append((color_bright, gray_img.copy(), "bright"))
    
    # 3. Mild Contrast (only 30% chance - most memory intensive)
    if random.random() > 0.7 and len(augmentations) < 3:  # Limit to 3 total
        # Small contrast adjustment
        alpha = random.uniform(0.95, 1.05)
        color_contrast = cv2.convertScaleAbs(color_img, alpha=alpha, beta=0)
        augmentations.append((color_contrast, gray_img.copy(), "contrast"))
    
    print(f"   Generated {len(augmentations)} augmented versions (including original)")
    return augmentations

# ---------------------------
# SHARED MOBILENET MODEL (Memory Optimization)
# ---------------------------
print("\n🤖 Initializing SHARED MobileNetV2 Model...")

class SharedMobileNet:
    """Single MobileNetV2 model shared between validator and feature extractor"""
    
    def __init__(self):
        print("   🔧 Loading SINGLE MobileNetV2 model...")
        try:
            # Load ONE MobileNetV2 model (17MB instead of 34MB)
            self.base_model = tf.keras.applications.MobileNetV2(
                weights="imagenet",
                input_shape=(224, 224, 3),
                include_top=True  # For classification
            )
            
            # Create feature extractor from the same model (without top layer)
            self.feature_model = tf.keras.Model(
                inputs=self.base_model.input,
                outputs=self.base_model.layers[-2].output  # Layer before classification
            )
            
            # ImageNet class indices for cats and dogs
            self.cat_classes = {281, 282, 283, 284, 285, 286, 287, 288, 289}
            self.dog_classes = set(range(151, 269))
            
            print("   ✅ SINGLE MobileNetV2 loaded (saves 17MB RAM)")
            print(f"   📊 Recognizes {len(self.cat_classes)} cat breeds")
            print(f"   📊 Recognizes {len(self.dog_classes)} dog breeds")
            
        except Exception as e:
            print(f"   ❌ Failed to load MobileNetV2: {e}")
            self.base_model = None
            self.feature_model = None
    
    def validate_pet(self, image_array, min_confidence=0.15):
        """Check if image is a cat or dog"""
        if self.base_model is None:
            return False, "Model not initialized", {"error": "Model not loaded"}
        
        try:
            # Preprocess
            img = cv2.resize(image_array, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.base_model.predict(img, verbose=0)[0]
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            # Check each top prediction
            for idx in top_indices:
                confidence = float(predictions[idx])
                
                if idx in self.cat_classes and confidence > min_confidence:
                    breed = self._get_breed_name(idx)
                    return True, f"Cat detected ({confidence:.2%})", {
                        "class": "cat",
                        "confidence": float(confidence),
                        "breed": breed
                    }
                
                if idx in self.dog_classes and confidence > min_confidence:
                    breed = self._get_breed_name(idx)
                    return True, f"Dog detected ({confidence:.2%})", {
                        "class": "dog",
                        "confidence": float(confidence),
                        "breed": breed
                    }
            
            # If no cat/dog found
            top_idx = int(top_indices[0])
            top_conf = float(predictions[top_idx])
            
            # Get readable class name
            from tensorflow.keras.applications.imagenet_utils import decode_predictions
            decoded = decode_predictions(np.expand_dims(predictions, axis=0), top=1)[0][0]
            class_name = decoded[1]
            
            return False, f"Not a cat or dog. Detected: {class_name} ({top_conf:.2%})", {
                "detected_class": class_name,
                "confidence": float(top_conf)
            }
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", {"error": str(e)}
    
    def extract_features(self, image_np: np.ndarray) -> np.ndarray:
        """Extract 1280-dimensional features"""
        if self.feature_model is None:
            return np.zeros(1280)
        
        try:
            # Convert to RGB if needed
            if len(image_np.shape) == 2:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Resize to 224x224
            image_resized = cv2.resize(image_rgb, (224, 224))
            
            # Normalize (ImageNet stats)
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Expand dimensions for batch
            input_data = np.expand_dims(image_normalized, axis=0)
            
            # Extract features
            features = self.feature_model.predict(input_data, verbose=0)
            features = features.flatten()
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ MobileNet feature extraction error: {e}")
            return np.zeros(1280)
    
    def _get_breed_name(self, class_idx):
        """Get breed name from class index"""
        try:
            from tensorflow.keras.applications.imagenet_utils import decode_predictions
            dummy_pred = np.zeros(1000)
            dummy_pred[class_idx] = 1.0
            decoded = decode_predictions(np.expand_dims(dummy_pred, axis=0), top=1)[0][0]
            return decoded[1]
        except:
            return f"class_{class_idx}"

# Initialize SINGLE shared MobileNet model (saves 17MB)
shared_mobilenet = SharedMobileNet()

# ---------------------------
# ORB Feature Detector
# ---------------------------
print("\n🎯 Setting up ORB detector...")
try:
    # Reduced features for Railway memory
    orb = cv2.ORB_create(
        nfeatures=300,  # Reduced from 500 to save memory
        scaleFactor=1.2,
        nlevels=3,      # Reduced from 4
        edgeThreshold=15,
        patchSize=20
    )
    print("   ✅ ORB detector ready (300 features for memory efficiency)")
except Exception as e:
    print(f"   ❌ Failed to initialize ORB: {e}")
    orb = None

# ---------------------------
# Facial Recognition
# ---------------------------
print("\n👁️ Setting up facial recognition...")

class FacialRecognition:
    def __init__(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("   ⚠️ Could not load face cascade")
                self.face_cascade = None
            else:
                print("   ✅ Face cascade loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize facial recognition: {e}")
            self.face_cascade = None
    
    def detect_face(self, image: np.ndarray) -> List[tuple]:
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scale_factor = 0.5
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
            
            faces = self.face_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back
            faces = [(int(x/scale_factor), int(y/scale_factor), 
                     int(w/scale_factor), int(h/scale_factor)) 
                    for (x, y, w, h) in faces]
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                return [faces[0]]
            
            return []
            
        except Exception as e:
            print(f"   ⚠️ Face detection error: {e}")
            return []
    
    def extract_facial_features(self, image: np.ndarray) -> np.ndarray:
        try:
            faces = self.detect_face(image)
            
            if len(faces) == 0:
                return np.zeros(500)  # Reduced size for memory
            
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return np.zeros(500)
            
            face_resized = cv2.resize(face_roi, (48, 48))  # Reduced from 64x64
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Smaller HOG features for memory
            win_size = (48, 48)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(face_gray).flatten()
            
            # Reduce dimensionality for memory
            if len(features) > 500:
                features = features[:500]
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Facial feature extraction error: {e}")
            return np.zeros(500)
    
    def compute_facial_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        try:
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return max(0.0, (similarity + 1) / 2)
            
        except Exception as e:
            print(f"   ⚠️ Facial similarity computation error: {e}")
            return 0.0

face_rec = FacialRecognition()

# ---------------------------
# Firebase Initialization
# ---------------------------
def init_firebase():
    print("\n🔥 Initializing Firebase...")
    
    firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
    bucket_name = os.environ.get("FIREBASE_STORAGE_BUCKET")
    
    if not firebase_creds_json:
        print("❌ FIREBASE_CREDENTIALS not set")
        return None
    if not bucket_name:
        print("❌ FIREBASE_STORAGE_BUCKET not set")
        return None
    
    try:
        creds_dict = json.loads(firebase_creds_json)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(creds_dict, temp_file)
            temp_path = temp_file.name
        
        cred = credentials.Certificate(temp_path)
        firebase_admin.initialize_app(cred, {
            "storageBucket": bucket_name
        })
        
        bucket = storage.bucket()
        os.unlink(temp_path)
        
        print(f"✅ Firebase initialized: {bucket.name}")
        return bucket
        
    except Exception as e:
        print(f"❌ Firebase error: {e}")
        return None

bucket = init_firebase()

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_image(image_bytes, is_color=False, apply_clahe=True):
    """Fixed: Added optional apply_clahe parameter"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        if is_color:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Only apply CLAHE to grayscale images when requested
        if not is_color and apply_clahe:
            # Lighter CLAHE for memory efficiency
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))  # Reduced from 1.5
            img = clahe.apply(img)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None

def extract_orb_features(image):
    if orb is None:
        return [], None
    
    try:
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return keypoints, descriptors
    except Exception as e:
        print(f"❌ ORB feature extraction error: {e}")
        return [], None

def match_orb_features(desc1, desc2, kp1, kp2):
    if desc1 is None or desc2 is None:
        return 0.0
    
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        if len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        
        score = len(good) / min(len(kp1), len(kp2))
        return min(score, 1.0)
        
    except Exception as e:
        print(f"❌ ORB matching error: {e}")
        return 0.0

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return max(0.0, (similarity + 1) / 2)

def hybrid_match_score(orb_score, mobile_score, face_score, 
                      mobilenet_weight, orb_weight, face_weight):
    return (orb_weight * orb_score) + (mobilenet_weight * mobile_score) + (face_weight * face_score)

# ---------------------------
# Database Management WITH AUGMENTATION
# ---------------------------
database = {}

def extract_features_from_augmentations(color_img, gray_img):
    """
    Extract features from original + augmented versions
    Returns: List of feature dictionaries
    """
    features_list = []
    augmentations = apply_light_augmentation(color_img, gray_img)
    
    for color_aug, gray_aug, aug_name in augmentations:
        try:
            # Extract ORB features from grayscale augmentation
            orb_kp, orb_desc = extract_orb_features(gray_aug)
            
            # Extract MobileNet features from color augmentation using shared model
            mobile_features = shared_mobilenet.extract_features(color_aug)
            
            # Extract facial features from color augmentation
            facial_features = face_rec.extract_facial_features(color_aug)
            
            features_list.append({
                'augmentation': aug_name,
                'orb_keypoints': orb_kp,
                'orb_descriptors': orb_desc,
                'mobile_features': mobile_features,
                'facial_features': facial_features,
                'is_augmented': aug_name != "original"
            })
            
        except Exception as e:
            print(f"   ⚠️ Failed to extract features for augmentation '{aug_name}': {e}")
            # Continue with other augmentations
    
    return features_list

def load_database():
    """Load database from Firebase with light augmentation"""
    global database
    print("\n🏗️ Building database with light augmentation...")
    
    if bucket is None:
        print("⚠️ Skipping database - Firebase not available")
        return
    
    try:
        blobs = list(bucket.list_blobs(prefix="noseprints/"))
        print(f"📁 Found {len(blobs)} files in Firebase")
        
        processed_count = 0
        augmented_count = 0
        failed_count = 0
        
        for blob in blobs:
            filename = blob.name.lower()
            if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
                continue
            
            parts = blob.name.split('/')
            if len(parts) < 3:
                continue
            
            pet_id = parts[1]
            
            print(f"  📥 Processing: {pet_id}")
            
            try:
                image_bytes = blob.download_as_bytes()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img_color is None:
                    print(f"  ❌ Failed to decode image")
                    failed_count += 1
                    continue
                
                img_gray = preprocess_image(image_bytes, is_color=False)
                if img_gray is None:
                    failed_count += 1
                    continue
                
                # Extract features from original + augmented versions
                all_features = extract_features_from_augmentations(img_color, img_gray)
                
                if not all_features:
                    print(f"  ⚠️ No features extracted for {pet_id}")
                    failed_count += 1
                    continue
                
                if pet_id not in database:
                    database[pet_id] = []
                
                for features in all_features:
                    database[pet_id].append({
                        'orb_keypoints': features['orb_keypoints'],
                        'orb_descriptors': features['orb_descriptors'],
                        'mobile_features': features['mobile_features'],
                        'facial_features': features['facial_features'],
                        'image_path': f"{blob.name}_{features['augmentation']}",
                        'is_augmented': features['is_augmented']
                    })
                    
                    if features['is_augmented']:
                        augmented_count += 1
                
                processed_count += 1
                
                # Memory management: Clear TensorFlow session periodically
                if processed_count % 10 == 0:
                    tf.keras.backend.clear_session()
                    print(f"  🔄 Cleared TensorFlow session for memory management")
                    
            except Exception as e:
                print(f"  ❌ Error processing {pet_id}: {e}")
                failed_count += 1
                continue
        
        print(f"\n✅ Database loaded successfully with augmentation!")
        print(f"   Successfully processed: {processed_count} images")
        print(f"   Failed to process: {failed_count} images")
        print(f"   Total pets in database: {len(database)}")
        
        if len(database) > 0:
            total_entries = sum(len(entries) for entries in database.values())
            print(f"   Total entries (including augmented): {total_entries}")
            print(f"   Augmented versions created: {augmented_count}")
            if processed_count > 0:
                print(f"   Average augmentations per image: {augmented_count/processed_count:.1f}")
        
        # Memory status
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            print(f"   💾 Current memory usage: {memory_usage:.1f}%")
        except:
            print(f"   💾 Memory info not available")
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        traceback.print_exc()

load_database()

# ---------------------------
# FastAPI Application
# ---------------------------
print("\n🚀 Creating FastAPI app...")
app = FastAPI(
    title="PawTag Hybrid Backend",
    description="Pet identification with light augmentation for Railway",
    version="4.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    total_entries = sum(len(entries) for entries in database.values()) if database else 0
    return {
        "service": "PawTag Hybrid Backend",
        "status": "online",
        "version": "4.4.0",
        "features": [
            "Cat/Dog Validator (MobileNetV2)",
            "MobileNetV2 Feature Extraction", 
            "ORB Feature Matching",
            "Facial Recognition",
            "Light Data Augmentation",
            "Improved Pet Rejection System"
        ],
        "database": {
            "pets_count": len(database),
            "total_entries": total_entries,
            "has_augmentation": True
        },
        "rejection_system": {
            "active": True,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "database_threshold": MIN_MATCH_SCORE_FOR_DB,
            "min_acceptable_score": MIN_ACCEPTABLE_SCORE
        },
        "memory_optimized": True,
        "platform": "Railway"
    }

@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    min_score: float = Query(0.33, description="Minimum confidence score"),
    strict_validation: bool = Query(False, description="Use strict cat/dog validation"),
    debug: bool = Query(False, description="Return detailed validation info")
):
    """
    Identify pet from image - Now with Cat/Dog validation and augmentation support
    """
    try:
        print(f"\n🔍 Identification request: {file.filename}")
        print(f"   Min score: {min_score}")
        print(f"   Strict validation: {strict_validation}")
        print(f"   Debug mode: {debug}")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_color is None:
            return JSONResponse(
                status_code=400, 
                content={
                    "success": False,
                    "error": "INVALID_IMAGE",
                    "message": "Invalid image file format",
                    "suggestion": "Please upload a valid JPG, PNG, or JPEG image"
                }
            )
        
        # Validate if it's a cat or dog
        print("   🐱🐕 Validating if image is a cat or dog...")
        is_cat_or_dog, cat_dog_message, cat_dog_details = shared_mobilenet.validate_pet(
            img_color,
            min_confidence=0.2 if strict_validation else 0.1
        )
        
        print(f"   Cat/Dog validation: {'✅ PASS' if is_cat_or_dog else '❌ FAIL'}")
        print(f"   Validation message: {cat_dog_message}")
        
        if not is_cat_or_dog:
            response = {
                "success": False,
                "error": "NOT_CAT_OR_DOG",
                "message": cat_dog_message,
                "suggestion": "Please upload a clear photo of a cat or dog. Other animals, humans, and objects are not accepted."
            }
            
            if debug:
                response["validation_details"] = convert_to_serializable(cat_dog_details)
            
            return JSONResponse(content=convert_to_serializable(response))
        
        # Only proceed if it's a valid cat/dog
        print("   ✅ Valid cat/dog detected. Proceeding with identification...")
        
        img_gray = preprocess_image(contents, is_color=False)
        if img_gray is None:
            return JSONResponse(
                status_code=400, 
                content=convert_to_serializable({
                    "success": False,
                    "error": "PROCESSING_ERROR",
                    "message": "Failed to process image"
                })
            )
        
        # Extract features from query image
        orb_kp, orb_desc = extract_orb_features(img_gray)
        mobile_features = shared_mobilenet.extract_features(img_color)
        facial_features = face_rec.extract_facial_features(img_color)
        
        print(f"   Feature extraction:")
        print(f"     - ORB features: {len(orb_kp) if orb_kp else 0} keypoints")
        print(f"     - MobileNet: {len(mobile_features)} features")
        print(f"     - Facial: {len(facial_features)} features")
        
        # Find matches against augmented database
        matches = []
        scores_by_pet = {}  # Track best scores per pet
        
        for pet_id, entries in database.items():
            pet_best_score = 0.0
            
            for entry in entries:
                # Calculate ORB score
                orb_score = match_orb_features(orb_desc, entry['orb_descriptors'], 
                                              orb_kp, entry['orb_keypoints'])
                
                # Calculate MobileNet score
                mobile_score = cosine_similarity(mobile_features, entry['mobile_features'])
                
                # Calculate face score
                face_score = face_rec.compute_facial_similarity(facial_features, entry['facial_features'])
                
                # Combined score (weighted)
                combined_score = hybrid_match_score(
                    orb_score, mobile_score, face_score,
                    DEFAULT_MOBILENET_WEIGHT, DEFAULT_ORB_WEIGHT, DEFAULT_FACE_WEIGHT
                )
                
                if combined_score > pet_best_score:
                    pet_best_score = combined_score
            
            # Store the best score for this pet
            scores_by_pet[pet_id] = pet_best_score
            
            # Add to matches if above threshold
            if pet_best_score >= min_score:
                matches.append({
                    "pet_id": pet_id,
                    "score": float(pet_best_score),
                    "score_percent": f"{pet_best_score * 100:.2f}%"
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"   Found {len(matches)} matches above threshold")
        if matches:
            print(f"   Best match score: {matches[0]['score']:.2f}")
            print(f"   Second best score: {matches[1]['score'] if len(matches) > 1 else 0:.2f}")
        
        # IMPROVED: REJECTION LOGIC FOR NON-DATABASE PETS
        should_reject = False
        rejection_reason = ""
        rejection_details = {}
        
        if len(matches) == 0:
            # No matches at all
            should_reject = True
            rejection_reason = "NO_DATABASE_MATCH"
            rejection_details = {
                "best_score_found": 0.0,
                "threshold_required": float(min_score),
                "message": "No matching pets found in database"
            }
        else:
            top_match = matches[0]
            top_score = top_match["score"]
            
            # HIGH CONFIDENCE CHECK: If score is very high, accept immediately
            if top_score >= HIGH_CONFIDENCE_THRESHOLD:
                print(f"   ✅ High confidence match ({top_score:.2f} >= {HIGH_CONFIDENCE_THRESHOLD}) - Accepting")
                should_reject = False
            else:
                # PRIMARY CHECK: Is the top score high enough for database?
                if top_score < MIN_MATCH_SCORE_FOR_DB:
                    should_reject = True
                    rejection_reason = "LOW_CONFIDENCE_MATCH"
                    rejection_details = {
                        "best_score": float(top_score),
                        "database_threshold": float(MIN_MATCH_SCORE_FOR_DB),
                        "message": f"Best match ({top_score:.2f}) is below database acceptance threshold ({MIN_MATCH_SCORE_FOR_DB})"
                    }
                
                # UNCERTAINTY CHECK: Only apply if we have multiple matches AND score is borderline
                elif len(matches) > 1 and top_score < HIGH_CONFIDENCE_THRESHOLD:
                    second_score = matches[1]["score"]
                    score_diff = top_score - second_score
                    
                    # Only reject for uncertainty if:
                    # 1. Difference is very small AND
                    # 2. Top score is not already very high
                    if score_diff < UNCERTAIN_SCORE_DIFF and top_score < 0.65:
                        should_reject = True
                        rejection_reason = "UNCERTAIN_MATCH"
                        rejection_details = {
                            "top_score": float(top_score),
                            "second_score": float(second_score),
                            "difference": float(score_diff),
                            "required_difference": float(UNCERTAIN_SCORE_DIFF),
                            "message": f"Top matches are too close ({score_diff:.2f} difference) and confidence is borderline"
                        }
                    else:
                        print(f"   ✅ Acceptable match difference: {score_diff:.2f} (top score: {top_score:.2f})")
                
                # ABSOLUTE MINIMUM CHECK: Is the score just too low overall?
                if top_score < ABSOLUTE_MIN_SCORE:
                    should_reject = True
                    rejection_reason = "INSUFFICIENT_OVERALL_SCORE"
                    rejection_details = {
                        "score": float(top_score),
                        "absolute_minimum": float(ABSOLUTE_MIN_SCORE),
                        "message": f"Overall match confidence ({top_score:.2f}) is too low"
                    }
        
        # Handle rejection
        if should_reject:
            print(f"   ❌ Rejecting: {rejection_reason}")
            print(f"   📊 Details: {rejection_details['message']}")
            
            # Get top 3 scores for debugging
            top_scores_list = []
            for i, match in enumerate(matches[:3]):
                top_scores_list.append({
                    "rank": i + 1,
                    "pet_id": match["pet_id"],
                    "score": float(match["score"]),
                    "score_percent": f"{match['score'] * 100:.1f}%"
                })
            
            response = {
                "success": False,
                "error": "PET_NOT_IN_DATABASE",
                "message": "This pet is not registered in our database",
                "rejection_reason": rejection_reason,
                "rejection_details": rejection_details,
                "cat_dog_validated": True,
                "cat_dog_details": convert_to_serializable(cat_dog_details),
                "match_stats": {
                    "matches_found": len(matches),
                    "database_threshold_used": float(MIN_MATCH_SCORE_FOR_DB),
                    "high_confidence_threshold": float(HIGH_CONFIDENCE_THRESHOLD),
                    "top_matches": convert_to_serializable(top_scores_list) if debug else len(top_scores_list)
                },
                "suggestion": "This appears to be a cat/dog, but it's not registered in our system. Please register the pet first."
            }
            
            if debug:
                # Calculate average score for context
                if scores_by_pet:
                    scores_list = list(scores_by_pet.values())
                    avg_score = float(np.mean(scores_list))
                    median_score = float(np.median(scores_list))
                    max_score = float(max(scores_list))
                    
                    response["feature_scores"] = {
                        "orb_features": len(orb_kp) if orb_kp else 0,
                        "faces_detected": len(face_rec.detect_face(img_color)),
                        "score_distribution": {
                            "average": avg_score,
                            "median": median_score,
                            "maximum": max_score,
                            "pets_above_0.5": sum(1 for s in scores_by_pet.values() if s > 0.5)
                        }
                    }
            
            return JSONResponse(content=convert_to_serializable(response))
        
        # If not rejected, proceed with success response
        # Limit to top 3 matches for response
        top_matches = matches[:3]
        
        # Determine confidence level
        top_score = top_matches[0]["score"]
        if top_score >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "very_high"
        elif top_score >= 0.65:
            confidence_level = "high"
        elif top_score >= 0.55:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Prepare success response
        response = {
            "success": True,
            "message": f"Found confident match in database",
            "matches": convert_to_serializable(top_matches),
            "match_confidence": confidence_level,
            "cat_dog_validated": True,
            "cat_dog_details": convert_to_serializable(cat_dog_details),
            "database_has_augmentation": True,
            "rejection_check": {
                "passed": True,
                "top_score": float(top_matches[0]["score"]),
                "database_threshold": float(MIN_MATCH_SCORE_FOR_DB),
                "high_confidence_met": bool(top_score >= HIGH_CONFIDENCE_THRESHOLD)
            }
        }
        
        if debug:
            top_score_diff = float(matches[0]["score"] - matches[1]["score"]) if len(matches) > 1 else 1.0
            response["debug_info"] = {
                "total_matches": len(matches),
                "top_score_difference": top_score_diff,
                "uncertainty_threshold": float(UNCERTAIN_SCORE_DIFF),
                "feature_details": {
                    "orb_features": len(orb_kp) if orb_kp else 0,
                    "faces_detected": len(face_rec.detect_face(img_color)),
                    "mobile_features_count": len(mobile_features),
                    "facial_features_count": len(facial_features)
                }
            }
        
        return JSONResponse(content=convert_to_serializable(response))
        
    except Exception as e:
        print(f"❌ Error during identification: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content=convert_to_serializable({
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": str(e),
                "suggestion": "Server is under memory pressure. Try again or contact support."
            })
        )

@app.get("/health")
async def health_check():
    """Health check with memory information"""
    total_entries = sum(len(entries) for entries in database.values()) if database else 0
    
    health_data = {
        "status": "healthy",
        "database_loaded": bool(len(database) > 0),
        "pets_count": len(database),
        "total_database_entries": total_entries,
        "rejection_system": {
            "active": True,
            "high_confidence_threshold": float(HIGH_CONFIDENCE_THRESHOLD),
            "database_threshold": float(MIN_MATCH_SCORE_FOR_DB),
            "min_acceptable_score": float(MIN_ACCEPTABLE_SCORE),
            "absolute_min_score": float(ABSOLUTE_MIN_SCORE),
            "uncertain_threshold": float(UNCERTAIN_SCORE_DIFF),
            "logic": "High confidence (>0.75) always accepts. Otherwise checks thresholds."
        },
        "models": {
            "shared_mobilenet": bool(shared_mobilenet.base_model is not None),
            "orb": bool(orb is not None),
            "face_recognition": bool(face_rec.face_cascade is not None)
        },
        "augmentation": {
            "enabled": True,
            "type": "light (2-3x per image)",
            "memory_efficient": True
        }
    }
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        health_data["memory"] = {
            "total_mb": int(memory.total // (1024 * 1024)),
            "available_mb": int(memory.available // (1024 * 1024)),
            "percent_used": float(memory.percent),
            "railway_limit_mb": 512
        }
        health_data["optimizations"] = {
            "shared_mobilenet": True,
            "reduced_orb_features": True,
            "reduced_facial_features": True,
            "light_augmentation": True
        }
    except:
        health_data["memory"] = {"info": "Not available"}
    
    return JSONResponse(content=convert_to_serializable(health_data))

@app.get("/refresh")
async def refresh_db():
    """Refresh database with current augmentation settings"""
    try:
        global database
        database = {}
        load_database()
        
        total_entries = sum(len(entries) for entries in database.values()) if database else 0
        
        response = {
            "success": True,
            "message": "Database refreshed with light augmentation",
            "database_stats": {
                "pets_count": len(database),
                "total_entries": total_entries,
                "augmentation_enabled": True
            }
        }
        
        return JSONResponse(content=convert_to_serializable(response))
        
    except Exception as e:
        print(f"❌ Error refreshing database: {e}")
        return JSONResponse(
            status_code=500, 
            content=convert_to_serializable({
                "success": False,
                "error": "REFRESH_ERROR",
                "message": str(e)
            })
        )

@app.get("/augmentation-info")
async def augmentation_info():
    """Get information about the augmentation system"""
    response = {
        "augmentation_system": "Light Memory-Efficient Augmentation",
        "version": "4.4.0",
        "features": [
            "Horizontal flip (always)",
            "Mild brightness adjustment (50% chance)",
            "Mild contrast adjustment (30% chance)",
            "Total: 2-3x data per image"
        ],
        "memory_optimizations": [
            "Single shared MobileNetV2 model (saves 17MB)",
            "Reduced ORB features (300 instead of 500)",
            "Reduced facial features (500 instead of 1764)",
            "Selective augmentation (not all images get all augmentations)"
        ],
        "railway_compatibility": {
            "target_memory": "Under 450MB",
            "free_tier_limit": 512,
            "buffer_for_new_data": "~60MB"
        },
        "expected_improvement": "5-10% accuracy boost with augmentation"
    }
    
    return JSONResponse(content=convert_to_serializable(response))

@app.get("/memory-status")
async def memory_status():
    """Detailed memory status for debugging"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        
        response = {
            "system_memory": {
                "total_mb": int(memory.total // (1024 * 1024)),
                "available_mb": int(memory.available // (1024 * 1024)),
                "used_mb": int(memory.used // (1024 * 1024)),
                "percent_used": float(memory.percent)
            },
            "process_memory": {
                "rss_mb": int(process.memory_info().rss // (1024 * 1024)),
                "vms_mb": int(process.memory_info().vms // (1024 * 1024))
            },
            "railway_limits": {
                "free_tier_mb": 512,
                "current_usage_mb": int(process.memory_info().rss // (1024 * 1024)),
                "available_buffer_mb": 512 - int(process.memory_info().rss // (1024 * 1024))
            },
            "recommendation": "Keep usage under 450MB for new dataset uploads"
        }
        
        return JSONResponse(content=convert_to_serializable(response))
    except Exception as e:
        return JSONResponse(content=convert_to_serializable({
            "error": f"Could not get memory status: {str(e)}",
            "recommendation": "Install psutil package for memory monitoring"
        }))

@app.get("/rejection-settings")
async def get_rejection_settings():
    """Get current rejection threshold settings"""
    total_entries = sum(len(entries) for entries in database.values()) if database else 0
    avg_features = sum(len(entries) for entries in database.values()) / len(database) if database else 0
    
    response = {
        "rejection_system": "Active (Improved v4.4.0)",
        "thresholds": {
            "high_confidence_threshold": float(HIGH_CONFIDENCE_THRESHOLD),
            "min_match_for_database": float(MIN_MATCH_SCORE_FOR_DB),
            "min_acceptable_score": float(MIN_ACCEPTABLE_SCORE),
            "absolute_min_score": float(ABSOLUTE_MIN_SCORE),
            "uncertain_score_diff": float(UNCERTAIN_SCORE_DIFF),
            "logic_explanation": "High confidence (>0.75) always accepts. Otherwise checks other thresholds."
        },
        "database_stats": {
            "total_pets": len(database),
            "avg_features_per_pet": float(avg_features)
        }
    }
    
    return JSONResponse(content=convert_to_serializable(response))

print("\n" + "=" * 70)
print("✅ PawTag Backend v4.4.0 is ready!")
print(f"📊 Database: {len(database)} pets loaded (with light augmentation)")
total_entries = sum(len(entries) for entries in database.values()) if database else 0
print(f"📈 Total database entries: {total_entries}")
print(f"🔧 Features:")
print(f"   1. Cat/Dog Validator (Shared MobileNetV2)")
print(f"   2. ORB Feature Matching (300 features)")
print(f"   3. MobileNetV2 Deep Features (Shared model)")
print(f"   4. Facial Recognition (Memory optimized)")
print(f"   5. Light Data Augmentation (2-3x)")
print(f"   6. Improved Pet Rejection System (Smart thresholds)")
print(f"   7. JSON Serialization Fix (Numpy compatibility)")
print(f"🛡️ Improved rejection thresholds:")
print(f"   - High confidence: >{HIGH_CONFIDENCE_THRESHOLD} (always accepts)")
print(f"   - Database match: >{MIN_MATCH_SCORE_FOR_DB}")
print(f"   - Absolute minimum: >{ABSOLUTE_MIN_SCORE}")
print(f"   - Score difference: >{UNCERTAIN_SCORE_DIFF} (more lenient)")
print(f"💾 Memory Optimized for Railway Free Tier")
print(f"📡 Server ready!")
print("=" * 70)
