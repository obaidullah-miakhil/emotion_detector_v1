# 🎯 Emotion Detection Web App (Flask + PyTorch)

This project is a **Flask-based Emotion Detection Web App** that uses a webcam or uploaded image to detect human emotions in real time using a **PyTorch AlexNet model trained on FER-2013 dataset**.

---

## 🧩 Features
- 📸 Real-time emotion detection using webcam feed.
- 🖼️ Upload an image to analyze emotion.
- 🧠 Powered by a pre-trained AlexNet model on FER-2013.
- 🌐 Simple web interface using Flask.
- ⚙️ Cross-platform setup (macOS, Windows, Linux).

---

## 🧱 Folder Structure
```
emotion_detector_v1/
│
├── app.py
├── predict.py
├── model/
│   └── alexnet_fer2013_epoch15.pth
├── static/
│   ├── uploads/
│   └── css/, js/ (optional for styling)
└── templates/
    └── index.html
```

---

## 🖥️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/emotion_detector_v1.git
cd emotion_detector_v1
```

### 2️⃣ Create a Virtual Environment
**macOS/Linux:**
```bash
python3 -m venv emotion_env
source emotion_env/bin/activate
```

**Windows:**
```bash
python -m venv emotion_env
emotion_env\Scripts\activate
```

### 3️⃣ Install Dependencies
Before installing, ensure your Python version is **3.10–3.12** (PyTorch doesn’t yet support 3.13).

```bash
pip install -r requirements.txt
```

If you face NumPy or PyTorch errors on macOS, use:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2.0" --force-reinstall
```

---

## 🚀 Run the App
```bash
python app.py
```

Then open your browser and visit:
👉 **http://127.0.0.1:5000/**

---

## 📷 Webcam Access on macOS
If you get a “camera not authorized” error:
1. Go to **System Settings → Privacy & Security → Camera**
2. Enable camera access for **Terminal** or **VS Code**
3. Re-run the app

---

## 🧠 Emotion Labels
The model detects these 7 emotions:
```
['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
```

---

## 🧾 Troubleshooting

| Issue | Solution |
|--------|-----------|
| `torch not found` | Use the PyTorch CPU install command above |
| `numpy version conflict` | Run `pip install "numpy<2.0" --force-reinstall` |
| `Camera not detected` | Ensure camera permissions are allowed in system settings |
| `Model not found` | Make sure the `.pth` file is inside the `model/` folder |

---

## 👨‍💻 Author
**Obaidullah Miakhil**  
 AI & Data Science Reseacher  
📧 Email: Obaidullah.miakhil.khan@gmail.com

---

## 📜 License
This project is open-source under the **MIT License**.
