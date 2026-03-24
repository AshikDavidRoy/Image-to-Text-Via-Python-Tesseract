# Table OCR Tool (Lightweight, CPU-Friendly)

A minimal Python-based GUI application for extracting text from table images using Tesseract OCR, OpenCV, and Tkinter. Designed specifically for low-power systems with limited RAM and CPU.

---

## 🔧 Features

* Lightweight GUI (Tkinter, no heavy frameworks)
* Table-aware OCR (row reconstruction from word bounding boxes)
* Two modes:

  * **Fast OCR** → quick, low CPU usage
  * **Accurate OCR** → multi-pass preprocessing + best result selection
* Image preprocessing (resize, thresholding, deskew)
* Confidence-based OCR optimization
* Export extracted table data to CSV

---

## 📦 Requirements

### System (Fedora)

```bash
sudo dnf install tesseract python3-tkinter
```

Verify:

```bash
tesseract --list-langs
```

---

### Python (inside virtual environment)

```bash
pip install numpy pillow pytesseract opencv-python-headless
```

---

## ▶️ Running the App

```bash
source ~/envs/ocr/bin/activate
python text.py
```

---

## 🧠 How It Works

### 1. Image Loading

* Uses OpenCV (`cv2.imread`) in grayscale mode
* Avoids RGB to reduce memory usage (~3× smaller)

---

### 2. Preprocessing Pipeline

Multiple image variants are generated:

* Resize (normalize resolution for OCR)
* Median blur (noise reduction)
* Deskew (fix rotated text)
* CLAHE (contrast enhancement)
* Thresholding:

  * Otsu
  * Adaptive
  * Inverted

Why:

> Different preprocessing improves OCR depending on image quality

---

### 3. Multi-Pass OCR (Accurate Mode)

For each image variant:

* Runs Tesseract with multiple segmentation modes:

  * `--psm 6` → uniform text block
  * `--psm 4` → column-based layout
  * `--psm 11` → sparse text

Each result is scored based on:

* Average confidence
* Number of detected words

Best result is selected automatically.

---

### 4. Structured Text Extraction (Important Part)

Instead of raw text:

* Uses `pytesseract.image_to_data()`
* Extracts:

  * word text
  * bounding box (x, y, width, height)
  * confidence

---

### 5. Table Reconstruction

Core logic:

1. Compute word center positions (x-axis)
2. Cluster into columns
3. Group words by line (block + paragraph + line ID)
4. Assign each word to nearest column
5. Build rows → table structure

Result:

```
col1    col2    col3
data    data    data
```

---

### 6. CSV Export

* Saves reconstructed rows directly into `.csv`
* Compatible with Excel / LibreOffice

---

## ⚙️ Performance Design (Low-End Optimized)

| Technique            | Benefit                   |
| -------------------- | ------------------------- |
| Grayscale processing | Lower memory usage        |
| Resize before OCR    | Faster computation        |
| OpenCV operations    | C-optimized speed         |
| No deep learning     | Works on CPU-only systems |
| Threaded GUI         | UI stays responsive       |

---

## ⚖️ Fast vs Accurate Mode

| Mode     | Speed  | Accuracy | Use Case            |
| -------- | ------ | -------- | ------------------- |
| Fast     | High   | Medium   | Clean images        |
| Accurate | Medium | High     | Tables, noisy scans |

---

## 🐛 Common Issues

### 1. `No module named tkinter`

```bash
sudo dnf install python3-tkinter
```

---

### 2. `UnidentifiedImageError`

* File is not a real image
* Check:

```bash
file test.png
```

---

### 3. OCR Output is Wrong

Try:

* Use **Accurate OCR**
* Use higher resolution image
* Ensure good contrast (black text on white)

---

## 📉 Limitations

* Not a full table parser (no merged cell detection)
* Works best with:

  * clear grid lines OR
  * consistent spacing
* Complex layouts may need ML-based models (not used here due to resource limits)

---

## 🚀 Possible Improvements

* Cell-level segmentation using line detection
* Export to Excel (.xlsx)
* Batch processing
* Language support (multi-lang OCR)

---

## 🧩 Design

This tool prioritizes:

* **Low resource usage**
* **Deterministic behavior**
* **Classical computer vision over heavy ML**
* **Practical accuracy improvements without GPU**

---

## 📄 License

Free to use and modify.
