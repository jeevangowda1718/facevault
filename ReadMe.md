# 🔐 FaceVault

> Real-time face registration & recognition web app built with Flask, DeepFace, and OpenCV. Register via webcam — no dataset needed.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![DeepFace](https://img.shields.io/badge/DeepFace-Facenet-green?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- 📷 **Live webcam registration** — capture your face directly from the browser
- 🧠 **Auto-generates face embeddings** — no pre-existing dataset required
- ⚡ **Real-time recognition** — identifies registered faces instantly via webcam
- 💾 **Local storage** — all encodings saved in `data/encodings.pkl` (no cloud)
- 🗑️ **Manage identities** — view and delete registered faces from the UI
- 🪟 **Windows friendly** — uses DeepFace instead of dlib (no CMake required)

---

## 🖥️ Demo

| Home | Register | Detect |
|------|----------|--------|
| Two-button landing page | Enter name + webcam capture | Live bounding box + name label |

---

## 🗂️ Project Structure

```
facevault/
│
├── app.py                  # Flask backend — routes, camera, recognition logic
├── requirements.txt        # Python dependencies
│
├── templates/
│   ├── index.html          # Home page
│   ├── register.html       # Face registration page
│   └── detect.html         # Real-time recognition page
│
├── static/
│   ├── style.css           # Dark terminal UI styles
│   └── script.js           # Keyboard shortcuts & camera cleanup
│
└── data/
    └── encodings.pkl       # Auto-created on first registration
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Face Embeddings | DeepFace (Facenet model) |
| Face Detection | OpenCV Haar Cascade |
| Camera Streaming | MJPEG via Flask Response |
| Storage | Pickle (local `.pkl` file) |
| Frontend | HTML, CSS, Vanilla JS |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 – 3.11
- A working webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jeevangowda1718/facevault.git
cd facevault

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

### Open in browser

```
http://localhost:5000
```

> **Note:** The first time you register a face, DeepFace will automatically download the Facenet model (~90MB). This only happens once.

---

## 📖 How It Works

### Registration
1. Go to **Register Face**
2. Enter your name and click **Start Camera**
3. Click **Capture Face** — the system grabs 12 frames
4. DeepFace generates 128-dimensional Facenet embeddings
5. Embeddings are averaged and saved to `data/encodings.pkl`

### Recognition
1. Go to **Start Recognition**
2. Camera opens with live MJPEG stream
3. Each frame: OpenCV Haar cascade detects face locations
4. DeepFace generates embeddings for each detected face
5. Cosine similarity is computed against all stored encodings
6. If similarity > 0.70, the matched name is displayed above the face box

---

## 🧪 Dependencies

```txt
flask
opencv-python
deepface
numpy
tf-keras
```

No `dlib`, no `CMake`, no C++ compilation required.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not opening | Check another app isn't using the webcam |
| "No face detected" on capture | Improve lighting; face the camera directly |
| Slow first capture | DeepFace is loading the Facenet model — wait a moment |
| `ModuleNotFoundError: deepface` | Run `pip install deepface tf-keras` |
| Port 5000 in use | Change port in `app.py`: `port=5001` |

---

## 📁 Data & Privacy

All face data is stored **locally** in `data/encodings.pkl`. Nothing is sent to any server or cloud service. To reset all registered faces, simply delete `data/encodings.pkl`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Jeevan Gowda**
- GitHub: [@jeevangowda1718](https://github.com/jeevangowda1718)

---

> ⭐ If you found this useful, consider giving the repo a star!