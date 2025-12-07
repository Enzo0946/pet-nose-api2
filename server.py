# server.py - TensorFlow Lite Version (Railway Compatible) - WITH CAT/DOG VALIDATION
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
from typing import List
import urllib.request

print("=" * 70)
print("🚀 Starting PawTag Backend - TensorFlow Lite Version")
print("=" * 70)

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
# NEW: Simple Cat/Dog Validator using MobileNetV2
# ---------------------------
print("\n🐱🐕 Initializing Cat/Dog Validator...")

class SimplePetValidator:
    """Simple validator using MobileNetV2 pre-trained on ImageNet"""
    
    def __init__(self):
        print("   🔧 Loading MobileNetV2...")
        try:
            # Load MobileNetV2 (17MB - acceptable for Railway)
            self.model = tf.keras.applications.MobileNetV2(
                weights="imagenet",
                input_shape=(224, 224, 3)
            )
            
            # ImageNet class indices for cats and dogs
            self.cat_classes = {
                281, 282, 283, 284, 285, 286, 287, 288, 289  # Cats
            }
            
            self.dog_classes = set(range(151, 269))  # Dogs (151-268)
            
            print("   ✅ Cat/Dog Validator loaded (MobileNetV2)")
            print(f"   📊 Recognizes {len(self.cat_classes)} cat breeds")
            print(f"   📊 Recognizes {len(self.dog_classes)} dog breeds")
            
        except Exception as e:
            print(f"   ❌ Failed to load Cat/Dog Validator: {e}")
            self.model = None
    
    def is_valid_pet(self, image_array, min_confidence=0.15):
        """
        Check if image is a cat or dog
        Returns: (is_valid, message, details)
        """
        if self.model is None:
            return False, "Validator not initialized", {"error": "Model not loaded"}
        
        try:
            # Preprocess
            img = cv2.resize(image_array, (224, 224))
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.model.predict(img, verbose=0)[0]
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            # Check each top prediction
            for idx in top_indices:
                confidence = float(predictions[idx])
                
                if idx in self.cat_classes and confidence > min_confidence:
                    return True, f"Cat detected ({confidence:.2%})", {
                        "class": "cat",
                        "confidence": confidence,
                        "breed": self._get_breed_name(idx)
                    }
                
                if idx in self.dog_classes and confidence > min_confidence:
                    return True, f"Dog detected ({confidence:.2%})", {
                        "class": "dog",
                        "confidence": confidence,
                        "breed": self._get_breed_name(idx)
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
                "confidence": top_conf
            }
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", {"error": str(e)}
    
    def _get_breed_name(self, class_idx):
        """Get breed name from class index"""
        try:
            from tensorflow.keras.applications.imagenet_utils import decode_predictions
            # Create dummy predictions
            dummy_pred = np.zeros(1000)
            dummy_pred[class_idx] = 1.0
            decoded = decode_predictions(np.expand_dims(dummy_pred, axis=0), top=1)[0][0]
            return decoded[1]
        except:
            return f"class_{class_idx}"

cat_dog_validator = SimplePetValidator()

# ---------------------------
# MobileNet Feature Extractor using TensorFlow Lite
# ---------------------------
print("\n🤖 Initializing MobileNetV2 (TensorFlow Lite)...")

class TFLiteFeatureExtractor:
    def __init__(self):
        try:
            # Use MobileNetV2 from TensorFlow Hub (no download needed)
            import tensorflow_hub as hub
            
            print("   ⬇️ Loading MobileNetV2 from TensorFlow Hub...")
            
            # Load MobileNetV2 feature extractor
            self.model = tf.keras.Sequential([
                hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                              trainable=False)
            ])
            
            # Build model with input shape
            self.model.build([None, 224, 224, 3])
            
            print("   ✅ MobileNetV2 TensorFlow loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize TensorFlow model: {e}")
            print("   ⚠️ Trying fallback to local MobileNet...")
            self.model = self._load_local_mobilenet()
    
    def _load_local_mobilenet(self):
        """Fallback to local MobileNetV2"""
        try:
            print("   🔄 Loading local MobileNetV2...")
            model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
            print("   ✅ Local MobileNetV2 loaded")
            return model
        except Exception as e:
            print(f"   ❌ Failed to load local MobileNet: {e}")
            return None
    
    def extract_features(self, image_np: np.ndarray) -> np.ndarray:
        if self.model is None:
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
            features = self.model.predict(input_data, verbose=0)
            features = features.flatten()
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ MobileNet feature extraction error: {e}")
            return np.zeros(1280)

mobilenet_extractor = TFLiteFeatureExtractor()

# ---------------------------
# ORB Feature Detector
# ---------------------------
print("\n🎯 Setting up ORB detector...")
try:
    orb = cv2.ORB_create(
        nfeatures=500,
        scaleFactor=1.2,
        nlevels=4,
        edgeThreshold=15,
        patchSize=20
    )
    print("   ✅ ORB detector ready")
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
                return np.zeros(1764)
            
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return np.zeros(1764)
            
            face_resized = cv2.resize(face_roi, (64, 64))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # HOG features
            win_size = (64, 64)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog.compute(face_gray).flatten()
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"   ⚠️ Facial feature extraction error: {e}")
            return np.zeros(1764)
    
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
# Helper Functions - FIXED
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
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
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
# Database Management - FIXED
# ---------------------------
database = {}

def load_database():
    """Load database from Firebase"""
    global database
    print("\n🏗️ Building database...")
    
    if bucket is None:
        print("⚠️ Skipping database - Firebase not available")
        return
    
    try:
        blobs = list(bucket.list_blobs(prefix="noseprints/"))
        print(f"📁 Found {len(blobs)} files in Firebase")
        
        for blob in blobs:
            filename = blob.name.lower()
            if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
                continue
            
            parts = blob.name.split('/')
            if len(parts) < 3:
                continue
            
            pet_id = parts[1]
            
            print(f"  📥 Processing: {pet_id}")
            
            image_bytes = blob.download_as_bytes()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_color is None:
                print(f"  ❌ Failed to decode image")
                continue
            
            img_gray = preprocess_image(image_bytes, is_color=False)
            if img_gray is None:
                continue
            
            # FIXED: Remove apply_clahe parameter for color images
            img_color_processed = preprocess_image(image_bytes, is_color=True)
            
            # Extract all features
            orb_kp, orb_desc = extract_orb_features(img_gray)
            mobile_features = mobilenet_extractor.extract_features(img_color_processed)
            facial_features = face_rec.extract_facial_features(img_color_processed)
            
            if pet_id not in database:
                database[pet_id] = []
            
            database[pet_id].append({
                'orb_keypoints': orb_kp,
                'orb_descriptors': orb_desc,
                'mobile_features': mobile_features,
                'facial_features': facial_features,
                'image_path': blob.name,
            })
        
        print(f"\n✅ Database loaded successfully!")
        print(f"   Total pets: {len(database)}")
        total_images = sum(len(entries) for entries in database.values())
        print(f"   Total images: {total_images}")
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        traceback.print_exc()

load_database()

# ---------------------------
# FastAPI Application
# ---------------------------
print("🚀 Creating FastAPI app...")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "service": "PawTag Hybrid Backend",
        "status": "online",
        "version": "4.0.0",
        "features": [
            "Cat/Dog Validator (MobileNetV2)",
            "MobileNetV2 (TensorFlow)", 
            "ORB",
            "Facial Recognition"
        ],
        "database": {
            "pets_count": len(database),
        },
        "inference_engine": "TensorFlow"
    }

@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    min_score: float = Query(0.33, description="Minimum confidence score"),
    strict_validation: bool = Query(False, description="Use strict cat/dog validation"),
    debug: bool = Query(False, description="Return detailed validation info")
):
    """
    Identify pet from image - Now with Cat/Dog validation
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
        
        # NEW: First validate if it's a cat or dog
        print("   🐱🐕 Validating if image is a cat or dog...")
        is_cat_or_dog, cat_dog_message, cat_dog_details = cat_dog_validator.is_valid_pet(
            img_color,
            min_confidence=0.2 if strict_validation else 0.1  # Higher threshold for strict mode
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
                response["validation_details"] = cat_dog_details
            
            return response
        
        # Only proceed if it's a valid cat/dog
        print("   ✅ Valid cat/dog detected. Proceeding with identification...")
        
        img_gray = preprocess_image(contents, is_color=False)
        if img_gray is None:
            return JSONResponse(
                status_code=400, 
                content={
                    "success": False,
                    "error": "PROCESSING_ERROR",
                    "message": "Failed to process image"
                }
            )
        
        # Extract features for matching
        orb_kp, orb_desc = extract_orb_features(img_gray)
        faces = face_rec.detect_face(img_color)
        
        print(f"   Feature extraction:")
        print(f"     - ORB features: {len(orb_kp) if orb_kp else 0}")
        print(f"     - Faces detected: {len(faces)}")
        
        mobile_features = mobilenet_extractor.extract_features(img_color)
        facial_features = face_rec.extract_facial_features(img_color)
        
        print(f"     - MobileNet: {len(mobile_features)} features")
        print(f"     - Facial: {len(facial_features)} features")
        
        # Find matches
        matches = []
        
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
            
            # Add to matches if above threshold
            if pet_best_score >= min_score:
                matches.append({
                    "pet_id": pet_id,
                    "score": pet_best_score,
                    "score_percent": f"{pet_best_score * 100:.2f}%"
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"   Found {len(matches)} matches above threshold")
        
        if not matches:
            response = {
                "success": False,
                "message": "No matching pet found in database",
                "best_score": "0.00%",
                "threshold": f"{min_score * 100:.0f}%",
                "cat_dog_validated": True,
                "cat_dog_details": cat_dog_details,
                "suggestion": "This cat/dog is not registered in our system"
            }
            
            if debug:
                response["feature_details"] = {
                    "orb_features": len(orb_kp) if orb_kp else 0,
                    "faces_detected": len(faces)
                }
            
            return response
        
        # Prepare success response
        response = {
            "success": True,
            "message": f"Found {len(matches)} matching pet(s)",
            "matches": matches[:3],
            "cat_dog_validated": True,
            "cat_dog_details": cat_dog_details
        }
        
        if debug:
            response["feature_details"] = {
                "orb_features": len(orb_kp) if orb_kp else 0,
                "faces_detected": len(faces)
            }
        
        return response
        
    except Exception as e:
        print(f"❌ Error during identification: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={
                "success": False,
                "error": "INTERNAL_ERROR",
                "message": str(e)
            }
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": len(database) > 0,
        "pets_count": len(database),
        "models": {
            "cat_dog_validator": cat_dog_validator.model is not None,
            "mobilenet_tf": mobilenet_extractor.model is not None,
            "orb": orb is not None,
            "face_recognition": face_rec.face_cascade is not None
        }
    }

@app.get("/refresh")
async def refresh_db():
    """Refresh database"""
    try:
        global database
        database = {}
        load_database()
        
        return {
            "success": True,
            "message": "Database refreshed",
            "database_stats": {
                "pets_count": len(database),
            }
        }
        
    except Exception as e:
        print(f"❌ Error refreshing database: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "success": False,
                "error": "REFRESH_ERROR",
                "message": str(e)
            }
        )

@app.get("/validation-info")
async def validation_info():
    """Get information about the validation system"""
    return {
        "validation_system": "MobileNetV2 Cat/Dog Classifier",
        "version": "4.0.0",
        "features": [
            "Cat detection (9+ breeds)",
            "Dog detection (120+ breeds)",
            "Rejects other animals, humans, objects",
            "Uses ImageNet pre-trained knowledge"
        ],
        "validation_thresholds": {
            "strict_mode": "20% confidence",
            "lenient_mode": "10% confidence",
            "parameter": "strict_validation=true/false"
        },
        "api_usage": {
            "default": "/identify?min_score=0.33",
            "strict": "/identify?min_score=0.33&strict_validation=true",
            "debug": "/identify?min_score=0.33&debug=true"
        }
    }

print("\n" + "=" * 70)
print("✅ PawTag Backend is ready!")
print(f"📊 Database: {len(database)} pets loaded")
print(f"🔧 Features:")
print(f"   1. Cat/Dog Validator (MobileNetV2)")
print(f"   2. ORB Feature Matching")
print(f"   3. MobileNetV2 Deep Features")
print(f"   4. Facial Recognition")
print(f"📡 Server ready!")
print("=" * 70)
