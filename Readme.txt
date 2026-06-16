# 🎭 Face Mesh — MediaPipe + OpenCV

Real-time 468-point 3D face landmark detection using Google's MediaPipe and OpenCV.

---

## 📦 Setup in VS Code

### 1. Clone / open this folder in VS Code

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the project
```bash
python face.py
```

Press **Q** to quit the window.

---

## 🗂️ Project Structure

```
face-mesh/
├── face.py            # Main script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## ✨ Features

- **Face Tessellation** — full mesh of 468 landmarks (green lines)
- **Face Contours** — outline of face, eyes, lips, eyebrows (bright green)
- **Iris Tracking** — left & right iris detection (blue/red)
- **Selfie-mirrored** — flipped horizontally for natural feel
- Supports up to **1 face** at a time (change `max_num_faces` in `face.py`)

---

## ⚙️ Requirements

- Python 3.8 – 3.11
- Webcam
- VS Code (recommended) or any terminal