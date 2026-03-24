import os
import csv
import threading
from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
from tkinter import filedialog, messagebox


# If Tesseract is not on PATH, uncomment and adjust:
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


def safe_imread_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("OpenCV could not read the image. The file may be invalid or unsupported.")
    return img


def resize_for_ocr(gray: np.ndarray, min_side: int = 1000, max_side: int = 1800):
    h, w = gray.shape[:2]
    longest = max(h, w)

    if longest > max_side:
        scale = max_side / float(longest)
        interpolation = cv2.INTER_AREA
    elif longest < min_side:
        scale = min_side / float(longest)
        interpolation = cv2.INTER_CUBIC
    else:
        return gray

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(gray, (new_w, new_h), interpolation=interpolation)


def deskew(gray: np.ndarray):
    # Estimate skew from foreground pixels.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size < 2000:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.4:
        return gray

    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_variants(gray: np.ndarray):
    gray = resize_for_ocr(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = deskew(gray)

    variants = []

    # 1) Clean grayscale
    variants.append(("gray", gray))

    # 2) CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(("clahe", clahe_img))

    # 3) Otsu threshold
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants.append(("otsu", otsu))

    # 4) Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    variants.append(("adaptive", adaptive))

    # 5) Inverted Otsu, useful for some scans
    inv_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    variants.append(("inv_otsu", inv_otsu))

    return variants


def ocr_words(img: np.ndarray, psm: int):
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

    words = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < 0:
            continue

        words.append({
            "text": text,
            "conf": conf,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
            "word_num": int(data["word_num"][i]),
        })

    return words


def score_words(words):
    if not words:
        return -1e9
    confs = [w["conf"] for w in words if w["conf"] >= 0]
    if not confs:
        return -1e9
    avg_conf = sum(confs) / len(confs)
    return avg_conf + min(25.0, len(words) * 0.15)


def cluster_1d(values, threshold):
    values = sorted(values)
    if not values:
        return []

    clusters = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return [sum(c) / len(c) for c in clusters]


def nearest_index(centers, value):
    best_idx = 0
    best_dist = abs(centers[0] - value)
    for i in range(1, len(centers)):
        d = abs(centers[i] - value)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def reconstruct_table_like_rows(words):
    if not words:
        return []

    widths = [max(1, w["width"]) for w in words]
    med_width = float(np.median(widths)) if widths else 30.0
    threshold = max(20, int(med_width * 1.3))

    x_centers = [w["left"] + (w["width"] / 2.0) for w in words]
    column_centers = cluster_1d(x_centers, threshold=threshold)

    if not column_centers:
        column_centers = [x_centers[0]]

    # Group by Tesseract line identifiers.
    line_groups = defaultdict(list)
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])
        line_groups[key].append(w)

    ordered_lines = sorted(
        line_groups.items(),
        key=lambda kv: (
            min(item["top"] for item in kv[1]),
            min(item["left"] for item in kv[1]),
        ),
    )

    rows = []
    for _, line_words in ordered_lines:
        line_words = sorted(line_words, key=lambda w: w["left"])
        cells = [""] * len(column_centers)

        for w in line_words:
            x = w["left"] + (w["width"] / 2.0)
            idx = nearest_index(column_centers, x)
            if cells[idx]:
                cells[idx] += " " + w["text"]
            else:
                cells[idx] = w["text"]

        while cells and cells[-1] == "":
            cells.pop()

        if any(cells):
            rows.append(cells)

    return rows


def rows_to_text(rows):
    if not rows:
        return ""
    return "\n".join("\t".join(cell.strip() for cell in row) for row in rows)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def fast_ocr(gray):
    # Single pass; lower CPU cost.
    variants = preprocess_variants(gray)
    img = variants[0][1]  # cleaned grayscale
    words = ocr_words(img, psm=6)
    rows = reconstruct_table_like_rows(words)
    text = rows_to_text(rows)
    return rows, text, score_words(words)


def accurate_ocr(gray):
    # Multi-pass; higher accuracy, more CPU time.
    best = {
        "score": -1e9,
        "rows": [],
        "text": "",
        "variant": "",
        "psm": None,
    }

    psm_candidates = [6, 4, 11]
    variants = preprocess_variants(gray)

    for variant_name, img in variants:
        for psm in psm_candidates:
            words = ocr_words(img, psm=psm)
            score = score_words(words)
            if score > best["score"]:
                rows = reconstruct_table_like_rows(words)
                text = rows_to_text(rows)
                best.update({
                    "score": score,
                    "rows": rows,
                    "text": text,
                    "variant": variant_name,
                    "psm": psm,
                })

    return best["rows"], best["text"], best["score"], best["variant"], best["psm"]


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Table OCR Tool")
        self.root.geometry("980x720")

        self.image_path = None
        self.last_rows = []

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=10)

        tk.Button(top, text="Open Image", command=self.open_image).pack(side="left", padx=5)
        tk.Button(top, text="Fast OCR", command=self.run_fast).pack(side="left", padx=5)
        tk.Button(top, text="Accurate OCR", command=self.run_accurate).pack(side="left", padx=5)
        tk.Button(top, text="Save CSV", command=self.save_csv).pack(side="left", padx=5)
        tk.Button(top, text="Clear", command=self.clear_output).pack(side="left", padx=5)

        self.status = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=10)

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(body)
        left.pack(side="left", fill="y")

        self.preview_label = tk.Label(left, text="Preview", width=42, height=18, relief="groove")
        self.preview_label.pack()

        right = tk.Frame(body)
        right.pack(side="right", fill="both", expand=True)

        self.text = tk.Text(right, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)

        scroll_y = tk.Scrollbar(right, command=self.text.yview)
        scroll_y.pack(side="right", fill="y")
        self.text.config(yscrollcommand=scroll_y.set)

        self.preview_image = None

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            # Validate with Pillow for a clear error if the file is corrupt.
            with Image.open(path) as im:
                im.verify()

            self.image_path = path
            self.show_preview(path)
            self.status.set(f"Loaded: {os.path.basename(path)}")
            self.text.delete("1.0", tk.END)
            self.last_rows = []

        except (UnidentifiedImageError, Exception) as e:
            messagebox.showerror("Invalid image", f"Could not open image:\n{e}")

    def show_preview(self, path):
        with Image.open(path) as im:
            im.thumbnail((420, 420))
            self.preview_image = ImageTk.PhotoImage(im.copy())
        self.preview_label.config(image=self.preview_image, text="")

    def clear_output(self):
        self.text.delete("1.0", tk.END)
        self.status.set("Cleared.")
        self.last_rows = []

    def run_fast(self):
        self.run_ocr(mode="fast")

    def run_accurate(self):
        self.run_ocr(mode="accurate")

    def run_ocr(self, mode):
        if not self.image_path:
            messagebox.showwarning("No image", "Please open an image first.")
            return

        self.status.set(f"Running {mode} OCR...")
        self.text.delete("1.0", tk.END)

        thread = threading.Thread(target=self._ocr_worker, args=(mode,), daemon=True)
        thread.start()

    def _ocr_worker(self, mode):
        try:
            gray = safe_imread_gray(self.image_path)

            if mode == "fast":
                rows, text, score = fast_ocr(gray)
                meta = "fast"
            else:
                rows, text, score, variant, psm = accurate_ocr(gray)
                meta = f"accurate | variant={variant} | psm={psm}"

            self.root.after(0, self._update_result, rows, text, score, meta)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("OCR error", str(e)))
            self.root.after(0, lambda: self.status.set("Error."))

    def _update_result(self, rows, text, score, meta):
        self.last_rows = rows
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, text if text else "[No text detected]")
        self.status.set(f"Done. Score: {score:.1f} | Mode: {meta}")

    def save_csv(self):
        if not self.last_rows:
            messagebox.showinfo("Nothing to save", "Run OCR first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            write_csv(path, self.last_rows)
            self.status.set(f"Saved CSV: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"CSV saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
