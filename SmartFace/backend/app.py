from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
import io
import pickle
import os
from datetime import datetime
import json
import torch.nn.functional as F
import subprocess
import sys
import time

app = Flask(__name__)
CORS(app)

# 1. Load Face Detector

try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
    print("✓ MTCNN (Facenet-PyTorch) loaded successfully")
    face_detector = 'mtcnn'
except Exception as e:
    print(f"⚠ MTCNN not available: {e}")
    mtcnn = None
    face_detector = None


# 2. Device

device = torch.device('cpu')   # kalau nanti mau GPU ganti ke 'cuda'


# 3. Model Definition
class ResNet50Embedding(nn.Module):
    def __init__(self, embed_dim=512, p_drop=0.5):
        super(ResNet50Embedding, self).__init__()
        resnet = models.resnet50(weights=None)  # state_dict ckpt akan override weight ini

        in_features = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()            # buang fc bawaan

        self.backbone = resnet
        self.bn = nn.BatchNorm1d(in_features)   # 2048
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_features, embed_dim)  # 2048 -> 512

    def forward(self, x):
        x = self.backbone(x)   # [B, 2048]
        x = self.bn(x)         # [B, 2048]
        x = self.dropout(x)
        x = self.fc(x)         # [B, 512]
        return x

# 4. Load ArcFace checkpoint
MODEL_PATH = 'best_gacor.pth'   # <--- pakai .pth

model = None
arc_weight = None
idx_to_class_map = {}
num_classes = 0
IMG_SIZE = 224

try:
    ckpt = torch.load(MODEL_PATH, map_location=device)

    # --- Ambil info kelas ---
    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = ckpt.get("idx_to_class", {})
    num_classes = len(class_to_idx) if class_to_idx else 70

    # Normalisasi idx_to_class → dict idx:int -> label:str
    if isinstance(idx_to_class, list):
        idx_to_class_map = {i: lbl for i, lbl in enumerate(idx_to_class)}
    elif isinstance(idx_to_class, dict) and all(isinstance(k, int) for k in idx_to_class.keys()):
        idx_to_class_map = idx_to_class
    elif isinstance(idx_to_class, dict) and all(isinstance(v, int) for v in idx_to_class.values()):
        # kasus LABEL→INT
        idx_to_class_map = {v: k for k, v in idx_to_class.items()}
    else:
        idx_to_class_map = {i: f"class_{i}" for i in range(num_classes)}

    # --- ukuran gambar dari ckpt (kalau ada) ---
    IMG_SIZE = ckpt.get("img_size", 224)

    # --- Bangun model embedding dan load state_dict ---
    model = ResNet50Embedding(embed_dim=512, p_drop=0.5)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # --- Ambil weight ArcFace ---
    arc_state = ckpt["arc"]
    if isinstance(arc_state, dict) and "weight" in arc_state:
        arc_weight = arc_state["weight"]
    else:
        arc_weight = arc_state.weight
    arc_weight = arc_weight.to(device)

    print("✓ ArcFace checkpoint loaded successfully!")
    print(f"  Classes: {num_classes}")
    print(f"  Sample labels: {[idx_to_class_map[i] for i in list(idx_to_class_map.keys())[:5]]} ...")

except Exception as e:
    print(f"✗ Error loading model: {e}")
    print(f"  Make sure {MODEL_PATH} exists in the backend folder")


# Transform (HARUS sama dgn val_tfms)

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std =[0.5, 0.5, 0.5]),
])

# ==========================
# 6. Attendance storage
# ==========================
ATTENDANCE_FILE = 'attendance.json'

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        try:
            with open(ATTENDANCE_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception:
            return []
    return []

def save_attendance(attendance_list):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance_list, f, indent=2)


# Face Detection

def detect_and_crop_face(image_array):
    """Detect face using MTCNN (Facenet-PyTorch) and return cropped face"""
    if mtcnn is None:
        # Fallback: return center crop if MTCNN not available
        h, w = image_array.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return image_array[y1:y1+size, x1:x1+size], None
    
    # Convert BGR to RGB for MTCNN
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    # Detect faces with MTCNN
    boxes, probs = mtcnn.detect(image_pil)
    
    if boxes is None or len(boxes) == 0:
        return None, None
    
    # Get first detected face (highest confidence)
    box = boxes[0]
    x1, y1, x2, y2 = box.astype(int)
    
    # Safety clamp
    h, w = image_array.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    
    face_crop = image_array[y1:y2, x1:x2]
    bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
    
    return face_crop, bbox

# Prediction

def predict_identity(face_image):
    """Predict identity from face image (ArcFace ResNet50)"""
    if model is None or arc_weight is None or not idx_to_class_map:
        return []

    # Convert to PIL if ndarray
    if isinstance(face_image, np.ndarray):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)

    img_tensor = test_transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        SCALE = 20.0   # Scale Mirip Confidence
        emb = model(img_tensor)                    # [1, 512]
        emb_norm = F.normalize(emb, dim=1)
        w_norm   = F.normalize(arc_weight, dim=1)  # [C, 512]
        logits   = torch.matmul(emb_norm, w_norm.t()) * SCALE
        probabilities = torch.softmax(logits, dim=1)[0]

    # Top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)

    predictions = []
    for prob, idx in zip(top3_prob, top3_idx):
        idx_int = idx.item()
        label = idx_to_class_map.get(idx_int, f"class_{idx_int}")
        confidence = prob.item() * 100
        predictions.append({
            'label': label,
            'confidence': round(confidence, 2)
        })

    return predictions

# API ROUTES

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'face_detector': face_detector,
        'mtcnn_loaded': mtcnn is not None,
        'num_classes': num_classes
    })

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        
        image_data = data.get('image', '')
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        face_crop, bbox = detect_and_crop_face(image)
        
        if face_crop is None:
            return jsonify({'error': 'No face detected'}), 400
        
        predictions = predict_identity(face_crop)
        
        if not predictions:
            return jsonify({'error': 'Model not available'}), 500
        
        # Encode cropped face
        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Draw bbox only (no label text overlay)
        if bbox:
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 4)
        
        _, buffer = cv2.imencode('.jpg', image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'bbox': bbox,
            'face_image': f'data:image/jpeg;base64,{face_base64}',
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        
        label = data.get('label')
        confidence = data.get('confidence')
        image = data.get('image')
        
        if not label or confidence is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        attendance_record = {
            'id': len(load_attendance()) + 1,
            'label': label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'status': 'present',
            'image': image
        }
        
        attendance_list = load_attendance()
        
        today = datetime.now().strftime('%Y-%m-%d')
        already_marked = any(
            record['label'] == label and record['date'] == today 
            for record in attendance_list
        )
        
        if already_marked:
            return jsonify({
                'success': False,
                'message': f'{label} sudah absen hari ini'
            }), 400
        
        attendance_list.append(attendance_record)
        save_attendance(attendance_list)
        
        return jsonify({
            'success': True,
            'message': f'Absensi {label} berhasil dicatat',
            'record': attendance_record
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attendance', methods=['GET'])
def get_attendance():
    try:
        attendance_list = load_attendance()
        
        date_filter = request.args.get('date')
        if date_filter:
            attendance_list = [
                record for record in attendance_list 
                if record['date'] == date_filter
            ]
        
        return jsonify({
            'success': True,
            'data': attendance_list,
            'total': len(attendance_list)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/attendance/<int:id>', methods=['DELETE'])
def delete_attendance(id):
    try:
        attendance_list = load_attendance()
        attendance_list = [record for record in attendance_list if record['id'] != id]
        save_attendance(attendance_list)
        
        return jsonify({
            'success': True,
            'message': 'Attendance record deleted'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve static files from frontend build
frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'dist')
if not os.path.exists(frontend_dist):
    print("Mencoba membangun frontend...")
    frontend_dir = os.path.join(os.path.dirname(__file__), '..')
    try:
        # Coba dengan bun dulu, kalau gagal pakai npm
        subprocess.run(['bun', 'run', 'build'], cwd=frontend_dir, check=True)
        print("Frontend berhasil dibangun dengan bun.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
            print("Frontend berhasil dibangun dengan npm.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Build frontend gagal: {e}. Menggunakan folder root sebagai fallback.")
            frontend_dist = os.path.join(os.path.dirname(__file__), '..')

app.static_folder = frontend_dist

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    print("="*80)
    print(" STARTING FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*80)
    print(f"Model: {'✓ Loaded' if model else '✗ Not loaded'}")
    print(f"Face Detector: {'✓ MTCNN' if mtcnn else '✗ Not loaded'}")
    print(f"Classes: {num_classes}")
    print("="*80)
    
    app.run(debug=False, host='127.0.0.1', port=5000)
