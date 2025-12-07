# server.py - ENHANCED VERSION FOR RAILWAY DEPLOYMENT
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
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List, Optional
from datetime import datetime

print("=" * 70)
print("🚀 Starting PawTag Enhanced Backend - Railway Deployment")
print("=" * 70)

# Test imports
try:
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ OpenCV: {cv2.__version__}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# ---------------------------
# Configuration for Railway
# ---------------------------
DEFAULT_MOBILENET_WEIGHT = 0.8  # Reduced for Railway performance
DEFAULT_ORB_WEIGHT = 0.1        # Increased since ORB works well
DEFAULT_FACE_WEIGHT = 0.1
DEFAULT_THRESHOLD = 0.6
DEFAULT_USE_AUGMENTED = True
DEFAULT_AUGMENTATION_COUNT = 5  # Reduced for Railway memory

print(f"\n⚖️ Feature weights for Railway:")
print(f"   MobileNet: {DEFAULT_MOBILENET_WEIGHT}")
print(f"   ORB: {DEFAULT_ORB_WEIGHT}")
print(f"   Facial: {DEFAULT_FACE_WEIGHT}")

# ---------------------------
# MobileNet Feature Extractor (Lightweight)
# ---------------------------
print("\n🤖 Initializing MobileNetV2 (Lightweight)...")

class MobileNetFeatureExtractor:
    def __init__(self):
        try:
            # Use smaller model for Railway
            self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            # Remove classification layer, keep feature extractor
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            
            # Simplified preprocessing for speed
            self.preprocess = transforms.Compose([
                transforms.Resize(224),  # Smaller than original
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            # Use CPU on Railway (likely no GPU)
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            
            print(f"   ✅ MobileNetV2 loaded on {self.device}")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize MobileNet: {e}")
            self.model = None
    
    def extract_features(self, image_np: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(1280)
        
        try:
            # Convert to RGB
            if len(image_np.shape) == 2:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_batch)
            
            features_np = features.squeeze().cpu().numpy().flatten()
            # Normalize features
            norm = np.linalg.norm(features_np)
            if norm > 0:
                features_np = features_np / norm
            
            return features_np
            
        except Exception as e:
            print(f"   ⚠️ MobileNet feature extraction error: {e}")
            return np.zeros(1280)

mobilenet_extractor = MobileNetFeatureExtractor()

# ---------------------------
# ORB Feature Detector (Optimized for Railway)
# ---------------------------
print("\n🎯 Setting up ORB detector...")
try:
    # Optimized for Railway (fewer features for speed)
    orb = cv2.ORB_create(
        nfeatures=500,      # Reduced for Railway performance
        scaleFactor=1.2,
        nlevels=4,         # Fewer levels for speed
        edgeThreshold=15,
        patchSize=20
    )
    print("   ✅ ORB detector ready (Railway optimized)")
except Exception as e:
    print(f"   ❌ Failed to initialize ORB: {e}")
    orb = None

# ---------------------------
# Facial Recognition (Lightweight)
# ---------------------------
print("\n👁️ Setting up facial recognition...")

class FacialRecognition:
    def __init__(self):
        try:
            # Try to load face cascade
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
            # Resize for faster detection on Railway
            scale_factor = 0.5
            small_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
            
            faces = self.face_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.1,
                minNeighbors=3,  # Reduced for speed
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back to original size
            faces = [(int(x/scale_factor), int(y/scale_factor), 
                     int(w/scale_factor), int(h/scale_factor)) 
                    for (x, y, w, h) in faces]
            
            if len(faces) > 0:
                # Return only the largest face
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
                return np.zeros(1764)  # Smaller feature vector
            
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return np.zeros(1764)
            
            # Smaller face size for Railway
            face_resized = cv2.resize(face_roi, (64, 64))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Simpler HOG features for speed
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
# Firebase Initialization (Same as before)
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
# Helper Functions (Optimized)
# ---------------------------
def preprocess_image(image_bytes, is_color=False):
    """Optimized preprocessing for Railway"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        if is_color:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        if not is_color:
            # Simplified CLAHE for speed
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
    """Optimized matching for Railway"""
    if desc1 is None or desc2 is None:
        return 0.0
    
    try:
        # Simple BFMatcher for speed
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # Looser ratio for Railway
                good.append(m)
        
        if len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        
        score = len(good) / min(len(kp1), len(kp2))
        return min(score, 1.0)
        
    except Exception as e:
        print(f"❌ ORB matching error: {e}")
        return 0.0

def cosine_similarity(vec1, vec2):
    """Optimized cosine similarity"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return max(0.0, (similarity + 1) / 2)  # Normalize to 0-1

def hybrid_match_score(orb_score, mobile_score, face_score, 
                      mobilenet_weight, orb_weight, face_weight):
    """Weighted combination of all features"""
    return (orb_weight * orb_score) + (mobilenet_weight * mobile_score) + (face_weight * face_score)

# ---------------------------
# Simple Augmentation for Railway
# ---------------------------
def simple_augment_image(img):
    """Simple augmentation for Railway (fast and memory efficient)"""
    augmentations = []
    
    # 1. Original
    augmentations.append(img.copy())
    
    # 2. Horizontal flip
    augmentations.append(cv2.flip(img, 1))
    
    # 3. Brightness variation
    augmentations.append(cv2.convertScaleAbs(img, alpha=1.2, beta=20))
    
    # 4. Mild blur
    augmentations.append(cv2.GaussianBlur(img, (3, 3), 0.5))
    
    # 5. Small rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1.0)
    augmentations.append(cv2.warpAffine(img, M, (w, h)))
    
    return augmentations

# ---------------------------
# Database Management
# ---------------------------
database = {}
original_images_count = 0
augmented_images_count = 0

def load_database():
    """Load database with simple augmentation for Railway"""
    global database, original_images_count, augmented_images_count
    print("\n🏗️ Building database (Railway optimized)...")
    
    if bucket is None:
        print("⚠️ Skipping database - Firebase not available")
        return
    
    try:
        blobs = list(bucket.list_blobs(prefix="noseprints/"))
        print(f"📁 Found {len(blobs)} files in Firebase")
        
        original_images_count = 0
        augmented_images_count = 0
        
        for blob in blobs:
            filename = blob.name.lower()
            if not (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
                continue
            
            parts = blob.name.split('/')
            if len(parts) < 3:
                continue
            
            pet_id = parts[1]
            
            print(f"  📥 Processing: {pet_id} - {blob.name}")
            
            image_bytes = blob.download_as_bytes()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_color is None:
                print(f"  ❌ Failed to decode image")
                continue
            
            img_gray = preprocess_image(image_bytes, is_color=False)
            if img_gray is None:
                continue
            
            img_color_processed = preprocess_image(image_bytes, is_color=True, apply_clahe=False)
            
            # Process original image
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
                'is_augmented': False,
                'augmentation_type': 'original'
            })
            original_images_count += 1
            
            # Generate simple augmentations (limited for Railway)
            if DEFAULT_AUGMENTATION_COUNT > 0:
                aug_images = simple_augment_image(img_gray)
                
                # Limit augmentations
                for idx, aug_img in enumerate(aug_images[:DEFAULT_AUGMENTATION_COUNT]):
                    try:
                        # Process augmented image
                        orb_kp_aug, orb_desc_aug = extract_orb_features(aug_img)
                        
                        # Use same color image for MobileNet/Facial (or create color version)
                        if len(img_color.shape) == 3:
                            mobile_features_aug = mobilenet_extractor.extract_features(img_color)
                            facial_features_aug = face_rec.extract_facial_features(img_color)
                        else:
                            mobile_features_aug = np.zeros(1280)
                            facial_features_aug = np.zeros(1764)
                        
                        database[pet_id].append({
                            'orb_keypoints': orb_kp_aug,
                            'orb_descriptors': orb_desc_aug,
                            'mobile_features': mobile_features_aug,
                            'facial_features': facial_features_aug,
                            'image_path': f"{blob.name}_aug{idx+1}",
                            'is_augmented': True,
                            'augmentation_type': f'aug_{idx+1}'
                        })
                        augmented_images_count += 1
                        
                    except Exception as e:
                        print(f"  ⚠️ Failed to generate augmentation {idx+1}: {e}")
        
        print(f"\n✅ Database loaded successfully!")
        print(f"   Total pets: {len(database)}")
        print(f"   Original images: {original_images_count}")
        print(f"   Augmented images: {augmented_images_count}")
        print(f"   Total entries: {original_images_count + augmented_images_count}")
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        traceback.print_exc()

# Load database at startup
load_database()

# ---------------------------
# FastAPI Application
# ---------------------------
print("🚀 Creating FastAPI app...")
app = FastAPI()

# CORS middleware
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
        "service": "PawTag Enhanced Backend",
        "status": "online",
        "version": "2.0.0",
        "features": ["ORB", "MobileNet", "Facial Recognition"],
        "database": {
            "pets_count": len(database),
            "original_images": original_images_count,
            "augmented_images": augmented_images_count,
            "total_entries": original_images_count + augmented_images_count
        },
        "settings": {
            "default_threshold": DEFAULT_THRESHOLD,
            "default_weights": {
                "mobilenet": DEFAULT_MOBILENET_WEIGHT,
                "orb": DEFAULT_ORB_WEIGHT,
                "face": DEFAULT_FACE_WEIGHT
            },
            "augmentation_enabled": DEFAULT_AUGMENTATION_COUNT > 0
        }
    }

@app.post("/identify")
async def identify_pet(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, description="Minimum confidence score (0-1)"),
    mobilenet_weight: float = Query(DEFAULT_MOBILENET_WEIGHT, description="Weight for MobileNet features (0-1)"),
    orb_weight: float = Query(DEFAULT_ORB_WEIGHT, description="Weight for ORB features (0-1)"),
    face_weight: float = Query(DEFAULT_FACE_WEIGHT, description="Weight for facial features (0-1)"),
    use_augmented: bool = Query(DEFAULT_USE_AUGMENTED, description="Use augmented images in matching")
):
    """
    Identify pet from image using hybrid approach (ORB + MobileNet + Facial)
    """
    try:
        # Validate weights
        weight_sum = mobilenet_weight + orb_weight + face_weight
        if abs(weight_sum - 1.0) > 0.001:
            return JSONResponse(
                status_code=400,
                content={"error": "Weights must sum to 1.0"}
            )
        
        print(f"\n🔍 Identification started: {file.filename}")
        print(f"   Threshold: {threshold}, Weights: M={mobilenet_weight}, O={orb_weight}, F={face_weight}")
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_color is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
        
        img_gray = preprocess_image(contents, is_color=False)
        if img_gray is None:
            return JSONResponse(status_code=400, content={"error": "Failed to process image"})
        
        # Extract features from query image
        orb_kp, orb_desc = extract_orb_features(img_gray)
        mobile_features = mobilenet_extractor.extract_features(img_color)
        facial_features = face_rec.extract_facial_features(img_color)
        
        print(f"   Query features - ORB: {len(orb_kp) if orb_kp else 0} keypoints")
        
        # Find best match
        best_score = 0.0
        best_pet_id = None
        best_component_scores = {}
        
        for pet_id, entries in database.items():
            pet_best_score = 0.0
            pet_component_scores = {}
            
            for entry in entries:
                if not use_augmented and entry['is_augmented']:
                    continue
                
                # Calculate ORB score
                orb_score = 0.0
                if orb_weight > 0:
                    orb_score = match_orb_features(orb_desc, entry['orb_descriptors'], 
                                                  orb_kp, entry['orb_keypoints'])
                
                # Calculate MobileNet score
                mobile_score = 0.0
                if mobilenet_weight > 0:
                    mobile_score = cosine_similarity(mobile_features, entry['mobile_features'])
                
                # Calculate face score
                face_score = 0.0
                if face_weight > 0:
                    face_score = face_rec.compute_facial_similarity(facial_features, entry['facial_features'])
                
                # Calculate hybrid score
                hybrid_score = hybrid_match_score(orb_score, mobile_score, face_score,
                                                 mobilenet_weight, orb_weight, face_weight)
                
                if hybrid_score > pet_best_score:
                    pet_best_score = hybrid_score
                    pet_component_scores = {
                        "orb": round(orb_score, 3),
                        "mobilenet": round(mobile_score, 3),
                        "facial": round(face_score, 3)
                    }
            
            if pet_best_score > best_score:
                best_score = pet_best_score
                best_pet_id = pet_id
                best_component_scores = pet_component_scores
        
        print(f"   Best match: {best_pet_id} with score: {best_score:.3f}")
        
        # Prepare response
        if best_score >= threshold:
            response = {
                "success": True,
                "predicted_pet_id": best_pet_id,
                "confidence": round(best_score, 4),
                "confidence_percentage": f"{best_score * 100:.1f}%",
                "component_scores": best_component_scores,
                "message": f"Match found: {best_pet_id} (Confidence: {best_score * 100:.1f}%)"
            }
        else:
            response = {
                "success": False,
                "message": f"No match found above threshold {threshold * 100:.0f}%",
                "best_match": best_pet_id,
                "best_score": round(best_score, 4),
                "best_score_percentage": f"{best_score * 100:.1f}%",
                "component_scores": best_component_scores,
                "threshold_required": f"{threshold * 100:.0f}%"
            }
        
        print(f"✅ Identification complete")
        return response
        
    except Exception as e:
        print(f"❌ Error during identification: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/refresh")
async def refresh_db():
    """
    Refresh the database from Firebase
    """
    try:
        global database, original_images_count, augmented_images_count
        database = {}
        original_images_count = 0
        augmented_images_count = 0
        load_database()
        
        return {
            "success": True,
            "message": "Database refreshed",
            "database_stats": {
                "pets_count": len(database),
                "original_images": original_images_count,
                "augmented_images": augmented_images_count,
                "total_entries": original_images_count + augmented_images_count
            }
        }
        
    except Exception as e:
        print(f"❌ Error refreshing database: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_loaded": len(database) > 0,
        "pets_count": len(database),
        "total_images": original_images_count + augmented_images_count,
        "models": {
            "mobilenet": mobilenet_extractor.model is not None,
            "orb": orb is not None,
            "face_recognition": face_rec.face_cascade is not None
        }
    }

# Keep your existing evaluate endpoint for compatibility
@app.get("/evaluate")
async def evaluate(threshold: float = 0.3):
    print(f"📊 Evaluation request with threshold: {threshold}")
    
    # Simplified evaluation for Railway
    if not database:
        return {"error": "Database is empty"}
    
    # You can implement a simpler evaluation here
    # For now, return basic info
    return {
        "database_stats": {
            "pets": len(database),
            "total_images": sum(len(entries) for entries in database.values())
        },
        "threshold_used": threshold,
        "note": "Full evaluation requires more computation. Use /health for system status."
    }

print("\n" + "=" * 70)
print("✅ PawTag Enhanced Backend is ready on Railway!")
print(f"📊 Database: {len(database)} pets loaded")
print(f"🔄 Augmentation: {augmented_images_count} augmented images")
print("\n🌐 Endpoints:")
print("   POST /identify - Identify pet with hybrid features")
print("   GET  /refresh  - Refresh database")
print("   GET  /health   - Health check")
print("   GET  /evaluate - Basic evaluation")
print("\n📡 Server ready at: https://your-railway-app.up.railway.app")
print("=" * 70)
