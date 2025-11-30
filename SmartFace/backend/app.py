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

app = Flask(__name__)
CORS(app)

# Load Face Detection (MTCNN from facenet-pytorch)
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
    print("✓ MTCNN (Facenet-PyTorch) loaded successfully")
    face_detector = 'mtcnn'
except Exception as e:
    print(f"⚠ MTCNN not available: {e}")
    mtcnn = None
    face_detector = None

# Device
device = torch.device('cpu')

# Model Definition (sama dengan training)
class FineTunedResNet50(nn.Module):
    def __init__(self, num_classes=70):
        super(FineTunedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=False)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model = resnet
        
    def forward(self, x):
        return self.model(x)

# Load Model
MODEL_PATH = 'best_finetuned_resnet50.pth'  # Model sekarang di folder backend
model = None
label_encoder = None
num_classes = 70

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    num_classes = len(label_encoder.classes_)
    
    model = FineTunedResNet50(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Classes: {num_classes}")
    print(f"  Labels: {label_encoder.classes_[:5]}... (showing first 5)")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Attendance storage
ATTENDANCE_FILE = 'attendance.json'

def load_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_attendance(attendance_list):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(attendance_list, f, indent=2)

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
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Crop face
    face_crop = image_array[y1:y2, x1:x2]
    
    # Return crop and bbox
    bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
    
    return face_crop, bbox

def predict_identity(face_image):
    """Predict identity from face image"""
    if model is None or label_encoder is None:
        return []
    
    # Convert to PIL
    if isinstance(face_image, np.ndarray):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
    
    # Transform
    img_tensor = test_transform(face_image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    # Get top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    predictions = []
    for prob, idx in zip(top3_prob, top3_idx):
        label = label_encoder.inverse_transform([idx.item()])[0]
        confidence = prob.item() * 100
        predictions.append({
            'label': label,
            'confidence': round(confidence, 2)
        })
    
    return predictions

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
        
        # Get image from base64
        image_data = data.get('image', '')
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Detect and crop face
        face_crop, bbox = detect_and_crop_face(image)
        
        if face_crop is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Predict identity
        predictions = predict_identity(face_crop)
        
        if not predictions:
            return jsonify({'error': 'Model not available'}), 500
        
        # Encode cropped face to base64
        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Draw bbox on original image
        if bbox:
            # Convert to PIL Image for custom font
            from PIL import ImageDraw, ImageFont
            
            # Draw rectangle with cv2
            cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 4)
            
            # Convert to PIL for text with custom font
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            
            # Try to load Poppins font, fallback to default if not available
            label_text = predictions[0]['label']
            font_size = 40
            try:
                # Try to load Poppins Bold
                font = ImageFont.truetype("C:/Windows/Fonts/Poppins-Bold.ttf", font_size)
            except:
                try:
                    # Try alternative font paths or fallback
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    # Use default font
                    font = ImageFont.load_default()
            
            # Calculate text size and center position
            bbox_text = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            text_x = bbox['x1'] + (bbox['x2'] - bbox['x1'] - text_width) // 2
            text_y = bbox['y1'] - text_height - 30
            
            # Pastikan text tidak keluar dari frame
            if text_x < 5:
                text_x = 5
            if text_y < 5:
                text_y = bbox['y2'] + 10
            
            # Draw text with outline for better visibility
            outline_color = (0, 0, 0)
            text_color = (0, 255, 0)
            
            # Draw outline
            for adj_x in [-2, -1, 0, 1, 2]:
                for adj_y in [-2, -1, 0, 1, 2]:
                    draw.text((text_x + adj_x, text_y + adj_y), label_text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((text_x, text_y), label_text, font=font, fill=text_color)
            
            # Convert back to cv2 format
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Encode annotated image to base64
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
        
        # Create attendance record
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
        
        # Load existing attendance
        attendance_list = load_attendance()
        
        # Check if already marked today
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
        
        # Add and save
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
        
        # Filter by date if provided
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

if __name__ == '__main__':
    print("="*80)
    print("🚀 STARTING FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*80)
    print(f"Model: {'✓ Loaded' if model else '✗ Not loaded'}")
    print(f"Face Detector: {'✓ MTCNN' if mtcnn else '✗ Not loaded'}")
    print(f"Classes: {num_classes}")
    print("="*80)
    app.run(debug=True, host='0.0.0.0', port=5000)
