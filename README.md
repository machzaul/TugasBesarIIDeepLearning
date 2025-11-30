# SmartFace - Sistem Absensi Face Recognition

<div align="center">
  <h2>Tugas Besar II Deep Learning - Kelas RA</h2>
</div>

## 👥 Anggota Kelompok

| Nama                       | NIM       |
| -------------------------- | --------- |
| Elsa Elisa Yohana Sianturi | 122140135 |
| Sikah Nubuahtul Ilmi       | 122140208 |
| Machzaul Harmansyah        | 122140172 |

## 📋 Deskripsi Project

**SmartFace** adalah sistem absensi berbasis pengenalan wajah (_Face Recognition_) yang dibangun menggunakan teknologi Deep Learning. Sistem ini memanfaatkan model ResNet50 yang telah di-_fine-tune_ untuk mengenali wajah dan mencatat kehadiran secara otomatis.

Project ini dikembangkan sebagai tugas besar mata kuliah Deep Learning kelas RA, mengintegrasikan teknologi Computer Vision dengan sistem informasi kehadiran yang modern dan efisien.

##  Fitur Utama

- **Face Recognition Real-time**: Deteksi dan identifikasi wajah secara langsung melalui webcam
- **Multi-Prediction Display**: Menampilkan top-3 prediksi dengan confidence score
- **Attendance Tracking**: Pencatatan kehadiran otomatis dengan timestamp
- **Face Detection**: Menggunakan MTCNN untuk deteksi wajah yang akurat

##  Arsitektur Sistem

### Backend (Flask + PyTorch)

- **Deep Learning Model**: Fine-tuned ResNet50 dengan 70 classes
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Framework**: Flask REST API
- **Image Processing**: OpenCV, PIL

### Frontend (React + TypeScript)

- **Framework**: React 18 dengan Vite
- **Styling**: Tailwind CSS
- **Camera Access**: react-webcam
- **State Management**: React Hooks

##  Teknologi yang Digunakan

### Backend

- Python 3.x
- PyTorch 2.8.0
- Flask 3.1.0
- OpenCV 4.12.0
- MTCNN (facenet-pytorch)
- scikit-learn 1.7.1

### Frontend

- React 18.3.1
- TypeScript 5.8.3
- Vite 5.4.19
- Tailwind CSS 3.4.17
- React Router DOM 6.30.1

##  Instalasi

### Prasyarat

- Node.js (v18 atau lebih tinggi)
- Python 3.8+
- pip
- Webcam/kamera

### 1. Clone Repository

```bash
git clone https://github.com/elsaelisa09/TugasBesarIIDeepLearning.git
cd TugasBesarIIDeepLearning/SmartFace
```

### 2. Setup Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Pastikan file model tersedia di folder backend/
# - best_finetuned_resnet50.pth
# - resnet50_arcface_best2.pt
```

### 3. Setup Frontend

```bash
# Kembali ke root SmartFace directory
cd ..

# Install dependencies
npm install
```

##  Menjalankan Aplikasi

### 1. Jalankan Backend Server

```bash
cd backend
python app.py
```

Server akan berjalan di `http://localhost:5000`

### 2. Jalankan Frontend Development Server

```bash
# Di terminal baru, dari root SmartFace directory
npm run dev
```

Aplikasi akan tersedia di `http://localhost:8080`

##  Cara Menggunakan

1. **Buka Aplikasi**: Akses `http://localhost:8080` di browser
2. **Izinkan Akses Kamera**: Sistem akan meminta izin akses webcam
3. **Posisikan Wajah**: Arahkan wajah ke kamera, sistem akan mendeteksi otomatis
4. **Lihat Hasil Prediksi**:
   - Nama terdeteksi ditampilkan dengan bounding box hijau
   - Top-3 prediksi muncul di panel kanan dengan confidence score
5. **Catat Kehadiran**: Klik tombol "Tandai Kehadiran" untuk menyimpan absensi
6. **Verifikasi**: Sistem mencegah absensi ganda untuk hari yang sama

##  Struktur Project

```
TugasBesarIIDeepLearning/
├── README.md
└── SmartFace/
    ├── backend/
    │   ├── app.py                          # Flask REST API
    │   ├── requirements.txt                # Python dependencies
    │   ├── best_finetuned_resnet50.pth    # Model weights (tidak di-push)
    │   ├── resnet50_arcface_best2.pt      # Alternative model (tidak di-push)
    │   └── attendance.json                 # Data absensi
    ├── src/
    │   ├── components/
    │   │   ├── CameraView.tsx             # Komponen kamera
    │   │   ├── AttendanceStatus.tsx       # Status kehadiran
    │   │   └── ui/                        # UI components
    │   ├── pages/
    │   │   └── Index.tsx                  # Halaman utama
    │   └── main.tsx                       # Entry point
    ├── public/
    │   └── Scan.png                       # Logo
    └── package.json                       # Node dependencies
```

## Detail Model

### ResNet50 Fine-tuned Architecture

```
Input: 224x224x3 RGB Image
├── ResNet50 Base (pretrained)
│   ├── Layer 1-2: Frozen
│   ├── Layer 3: Fine-tuned
│   └── Layer 4: Fine-tuned
└── Custom FC Layers:
    ├── Dropout(0.5)
    ├── Linear(2048 → 512)
    ├── ReLU
    ├── BatchNorm1d(512)
    ├── Dropout(0.3)
    └── Linear(512 → 70)
```

### Training Details

- **Dataset**: 70 classes (identitas mahasiswa)
- **Image Size**: 224x224 pixels
- **Augmentation**: Random crop, horizontal flip, color jitter
- **Loss Function**: Cross-Entropy Loss

##  Referensi

1. **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. **MTCNN**: Zhang, K., et al. (2016). "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
3. **ArcFace**: Deng, J., et al. (2019). "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

##  Lisensi

Project ini dibuat untuk keperluan akademik Tugas Besar Deep Learning.

---

<div align="center">
  <p>Trio Kwek Kwek Deep Learning - Kelas RA</p>
  <p><strong>Institut Teknologi Sumatera</strong></p>
  <p>2025</p>
</div>
