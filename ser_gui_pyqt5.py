"""
Modern PyQt5 GUI for Speech Emotion Recognition — USING MODEL'S FEATURE EXTRACTOR
- Uses the same feature extraction + scaler as training
- Robust padding/truncation for variable frame feature lengths
- Consistent 3D scaling (matches training pipeline)
- Clean, modern UI with inline SVG icons
"""

import sys
import os
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QProgressBar, QFrame, QMessageBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QByteArray
from PyQt5.QtGui import QFont, QColor, QPainter, QPen
from PyQt5.QtSvg import QSvgWidget

# ================== TENSORFLOW / MODEL IMPORTS ==================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

MODEL_AVAILABLE = True
try:
    import tensorflow as tf
    # Best-effort: disable GPU to avoid unexpected device issues on some Windows setups
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

    # Import your project functions
    from ser.models.emotion_model import load_ser_model, load_scaler_and_classes
    from ser.features.feature_extractor import extended_extract_feature
    from ser.utils import read_audio_file
except Exception as e:
    MODEL_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ================== INFERENCE HELPERS (MATCH TRAINING) ==================
def _safe_input_tf(model) -> tuple:
    """
    Return (T_train_max, F_train_max) from model.input_shape.
    Works if input_shape is (None, T, F) or in a nested list.
    """
    in_shape = model.input_shape
    if isinstance(in_shape, (list, tuple)) and isinstance(in_shape[0], (list, tuple)):
        in_shape = in_shape[0]
    T_train_max = int(in_shape[-2])
    F_train_max = int(in_shape[-1])
    return T_train_max, F_train_max

def _pad_1d(vec: np.ndarray, target_F: int) -> np.ndarray:
    """Pad/truncate a single frame feature vector to target_F."""
    v = np.nan_to_num(np.asarray(vec).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if v.shape[0] == target_F:
        return v
    if v.shape[0] < target_F:
        out = np.zeros((target_F,), dtype=np.float32)
        out[:v.shape[0]] = v
        return out
    return v[:target_F]

def _pad_sequence_2d(seq: np.ndarray, max_T: int, max_F: int) -> np.ndarray:
    """Pad/truncate a (T, F) array to (max_T, max_F)."""
    out = np.zeros((max_T, max_F), dtype=np.float32)
    T = min(seq.shape[0], max_T)
    F = min(seq.shape[1], max_F)
    out[:T, :F] = seq[:T, :F]
    return out

def _transform_with_scaler_3d(X: np.ndarray, scaler) -> np.ndarray:
    """
    Apply StandardScaler fitted on flattened (N*T, F) to (N, T, F).
    This is exactly how training used it.
    """
    N, T, F = X.shape
    Xf = scaler.transform(X.reshape(N * T, F)).reshape(N, T, F)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xf

# ================== MODEL-COMPATIBLE PREDICTION ==================
def predict_emotions_compatible(file_path: str) -> dict:
    """
    Uses the same extractor + scaler as training. Returns a rich result dict:
    {
      file, duration_seconds, overall_emotion, confidence,
      all_emotion_scores: {emotion: prob, ...},
      timestamp: {start, end, emotion}
    }
    """
    if not MODEL_AVAILABLE:
        raise RuntimeError(f"Model not available: {IMPORT_ERROR}")

    # Load model + artifacts
    model = load_ser_model(best=True)
    scaler, class_names = load_scaler_and_classes()
    T_train_max, F_train_max = _safe_input_tf(model)

    # Extract frame-level features (list of 1D arrays; lengths may differ)
    frames = extended_extract_feature(file_path, frame_size=3, frame_stride=1, augment=False)
    if not frames:
        raise ValueError("No features extracted from audio.")

    # Normalize each frame to F_train_max, then stack (T, F)
    seq = np.stack([_pad_1d(f, F_train_max) for f in frames], axis=0)  # (T, F)
    # Pad/truncate time
    seq = _pad_sequence_2d(seq, max_T=T_train_max, max_F=F_train_max)  # (T, F)
    # Add batch dimension and scale like training
    seq = seq[None, ...]  # (1, T, F)
    seq = _transform_with_scaler_3d(seq, scaler)

    # Predict
    probs = model.predict(seq, verbose=0)[0]  # (C,)
    pred_idx = int(np.argmax(probs))
    emotion_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
    confidence = float(probs[pred_idx])

    # Build score dict
    all_scores = { (class_names[i] if i < len(class_names) else f"Class_{i}"): float(score)
                   for i, score in enumerate(probs) }

    # Duration
    audio, sr = read_audio_file(file_path)
    duration = float(len(audio)) / float(sr)

    return {
        "file": os.path.basename(file_path),
        "duration_seconds": round(duration, 2),
        "overall_emotion": emotion_name,
        "confidence": confidence,
        "all_emotion_scores": all_scores,
        "timestamp": {
            "start": 0.0,
            "end": round(duration, 2),
            "emotion": emotion_name,
        },
    }

# ================== AUDIO VALIDATION ==================
class AudioValidator:
    @staticmethod
    def validate_audio_file(file_path):
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist."

            file_size = os.path.getsize(file_path)
            if file_size < 10 * 1024:
                return False, f"File too small ({file_size/1024:.1f} KB). Minimum 10 KB required."

            valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in valid_extensions:
                return False, f"Unsupported format: {file_ext}. Supported: WAV, MP3, FLAC, OGG, M4A, AAC."

            return True, "File is valid."
        except Exception as e:
            return False, f"Error validating file: {str(e)}"

# ================== WORKER THREAD ==================
class EmotionWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            ok, msg = AudioValidator.validate_audio_file(self.file_path)
            if not ok:
                self.error.emit(msg)
                return

            results = predict_emotions_compatible(self.file_path)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}")

# ================== UI WIDGETS ==================
class CircularProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.maximum = 100
        self.progress_color = QColor("#4A90E2")
        self.setFixedSize(160, 160)

    def setValue(self, value):
        self.value = max(0, min(int(value), self.maximum))
        self.update()

    def setProgressColor(self, color):
        self.progress_color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background ring
        painter.setPen(QPen(QColor("#E9ECEF"), 10))
        painter.drawEllipse(15, 15, 130, 130)

        # Progress arc
        painter.setPen(QPen(self.progress_color, 10))
        span_angle = int(360 * self.value / self.maximum)
        painter.drawArc(15, 15, 130, 130, 90 * 16, -span_angle * 16)

        # Percentage text
        painter.setPen(QColor("#2C3E50"))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.value}%")

# Simple inline SVGs for emotions (clean + lightweight)
def emotion_svg_xml(emotion: str) -> str:
    color_map = {
        'angry': '#FF6B6B', 'calm': '#20B2AA', 'disgust': '#32CD32',
        'fearful': '#9370DB', 'happy': '#FFD700', 'neutral': '#4A90E2',
        'sad': '#6A5ACD', 'surprised': '#FFA500'
    }
    face = color_map.get(emotion.lower(), '#4A90E2')
    # Minimal face icon (circle + eyes + mouth variations)
    mouth = {
        'happy': 'M40 65 Q60 80 80 65',
        'sad': 'M40 75 Q60 60 80 75',
        'angry': 'M40 65 L80 65',
        'neutral': 'M40 70 L80 70',
        'calm': 'M40 68 Q60 72 80 68',
        'disgust': 'M40 70 Q60 75 80 70',
        'fearful': 'M40 68 Q60 85 80 68',
        'surprised': 'M56 65 A10 10 0 1 0 64 65 A10 10 0 1 0 56 65'
    }.get(emotion.lower(), 'M40 70 L80 70')

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="g" cx="50%" cy="35%" r="70%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="{face}"/>
    </radialGradient>
  </defs>
  <circle cx="60" cy="60" r="50" fill="url(#g)" stroke="#e5e7eb" stroke-width="2"/>
  <circle cx="45" cy="50" r="5" fill="#0f172a"/>
  <circle cx="75" cy="50" r="5" fill="#0f172a"/>
  <path d="{mouth}" stroke="#0f172a" stroke-width="4" fill="none" stroke-linecap="round"/>
</svg>"""

# ================== MAIN GUI ==================
class SERGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Emotion Recognition")
        self.setGeometry(100, 100, 1000, 720)
        self.setStyleSheet("QMainWindow { background-color: #F8F9FA; }")

        self.current_file = None
        self.worker = None

        self.results_section = None
        self.file_label = None
        self.analyze_btn = None
        self.emotion_label = None
        self.confidence_label = None
        self.circular_progress = None
        self.scores_grid = None
        self.svg_widget = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setSpacing(20)
        main.setContentsMargins(30, 30, 30, 30)

        header = QLabel("Speech Emotion Recognition")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("QLabel { font-size: 28px; font-weight: bold; color: #2C3E50; padding: 10px; }")
        header.setFont(QFont("Arial", 28, QFont.Bold))
        main.addWidget(header)

        status = QLabel(
            "✅ Model & extractor ready" if MODEL_AVAILABLE
            else f"⚠️ Model unavailable: {IMPORT_ERROR}"
        )
        status.setAlignment(Qt.AlignCenter)
        status.setStyleSheet(
            "QLabel { font-size: 12px; color: %s; padding: 6px; background-color: %s; border-radius: 6px; }" %
            (("#155724", "#D4EDDA") if MODEL_AVAILABLE else ("#856404", "#FFF3CD"))
        )
        main.addWidget(status)

        main.addWidget(self._file_section())
        main.addWidget(self._emotion_section())

        self.results_section = self._results_section()
        self.results_section.hide()
        main.addWidget(self.results_section)

        main.addWidget(self._requirements_section())

    # ---------- Sections ----------
    def _file_section(self):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; padding: 20px; border: 2px solid #E9ECEF; }")
        v = QVBoxLayout(frame)

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setStyleSheet("QLabel { font-size: 14px; color: #6C757D; padding: 8px; }")
        v.addWidget(self.file_label)

        fmt = QLabel("Supported formats: WAV, MP3, FLAC, OGG, M4A, AAC")
        fmt.setAlignment(Qt.AlignCenter)
        fmt.setStyleSheet("QLabel { font-size: 11px; color: #8E8E8E; padding: 4px; }")
        v.addWidget(fmt)

        h = QHBoxLayout()
        h.setAlignment(Qt.AlignCenter)

        browse = QPushButton("Browse Audio File")
        browse.setCursor(Qt.PointingHandCursor)
        browse.setStyleSheet(self._btn_style())
        browse.clicked.connect(self._browse_file)
        h.addWidget(browse)

        self.analyze_btn = QPushButton("Analyze Emotion")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setStyleSheet(self._btn_style(primary=True))
        self.analyze_btn.clicked.connect(self._analyze_emotion)
        self.analyze_btn.setEnabled(False)
        h.addWidget(self.analyze_btn)

        v.addLayout(h)
        return frame

    def _emotion_section(self):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; padding: 20px; border: 2px solid #E9ECEF; }")
        h = QHBoxLayout(frame)
        h.setAlignment(Qt.AlignCenter)
        h.setSpacing(30)

        # Progress + SVG face
        container = QVBoxLayout()
        container.setAlignment(Qt.AlignCenter)

        self.circular_progress = CircularProgressBar()
        # Place an SVG widget on top of progress, centered
        self.svg_widget = QSvgWidget()
        self.svg_widget.setFixedSize(120, 120)
        # Load a neutral face initially
        self._set_emotion_svg('neutral')

        svg_holder = QVBoxLayout()
        svg_holder.setContentsMargins(0, 0, 0, 0)
        svg_holder.setAlignment(Qt.AlignCenter)
        svg_holder.addWidget(self.svg_widget)

        # Stack the svg visually by just placing it under progress in the layout
        container.addWidget(self.circular_progress, alignment=Qt.AlignCenter)
        container.addLayout(svg_holder)

        h.addLayout(container)

        info = QVBoxLayout()
        info.setSpacing(10)
        info.setAlignment(Qt.AlignCenter)

        self.emotion_label = QLabel("Select a file to analyze")
        self.emotion_label.setAlignment(Qt.AlignCenter)
        self.emotion_label.setStyleSheet("QLabel { font-size: 24px; font-weight: bold; color: #2C3E50; padding: 6px; }")
        info.addWidget(self.emotion_label)

        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("QLabel { font-size: 16px; color: #6C757D; }")
        info.addWidget(self.confidence_label)

        h.addLayout(info)
        return frame

    def _results_section(self):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: white; border-radius: 12px; padding: 20px; border: 2px solid #E9ECEF; }")
        v = QVBoxLayout(frame)

        hdr = QLabel("Detailed Emotion Analysis")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; color: #2C3E50; padding: 8px; }")
        v.addWidget(hdr)

        self.scores_grid = QGridLayout()
        self.scores_grid.setSpacing(12)
        self.scores_grid.setContentsMargins(10, 10, 10, 10)
        v.addLayout(self.scores_grid)
        return frame

    def _requirements_section(self):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #E8F4FD; border-radius: 8px; padding: 12px; border: 1px solid #B8D4F0; }")
        v = QVBoxLayout(frame)

        txt = QLabel(
            "🎵 Model Requirements:\n"
            "• Uses same feature extraction as training\n"
            "• Auto padding/truncation to match (T, F)\n"
            "• Proper 3D scaling with trained scaler\n"
            "• Clean speech works best"
        )
        txt.setAlignment(Qt.AlignCenter)
        txt.setStyleSheet("QLabel { font-size: 11px; color: #2C3E50; padding: 4px; }")
        v.addWidget(txt)
        return frame

    # ---------- UI helpers ----------
    def _btn_style(self, primary=False):
        if primary:
            return """
                QPushButton {
                    background-color: #3B82F6; color: white; border: none;
                    padding: 12px 24px; border-radius: 10px; font-weight: bold; font-size: 14px;
                }
                QPushButton:hover { background-color: #2563EB; }
                QPushButton:pressed { background-color: #1D4ED8; }
                QPushButton:disabled { background-color: #93C5FD; color: #e5e7eb; }
            """
        return """
            QPushButton {
                background-color: white; color: #111827; border: 2px solid #E5E7EB;
                padding: 12px 24px; border-radius: 10px; font-weight: 600; font-size: 14px;
            }
            QPushButton:hover { background-color: #F3F4F6; }
            QPushButton:pressed { background-color: #E5E7EB; }
        """

    def _set_emotion_svg(self, emotion: str):
        xml = emotion_svg_xml(emotion)
        self.svg_widget.load(QByteArray(xml.encode('utf-8')))

    # ---------- Actions ----------
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All Files (*)"
        )
        if not path:
            return
        ok, msg = AudioValidator.validate_audio_file(path)
        if not ok:
            QMessageBox.warning(self, "Invalid Audio File", msg)
            return
        self.current_file = path
        name = os.path.basename(path)
        if len(name) > 40:
            name = name[:37] + "..."
        self.file_label.setText(f"Selected: {name}")
        self._reset_display()
        self.analyze_btn.setEnabled(True)

    def _reset_display(self):
        self.emotion_label.setText("Ready to analyze")
        self.confidence_label.setText("")
        self.circular_progress.setValue(0)
        self.circular_progress.setProgressColor("#4A90E2")
        self._set_emotion_svg('neutral')
        # Clear grid
        if self.scores_grid:
            for i in reversed(range(self.scores_grid.count())):
                w = self.scores_grid.itemAt(i).widget()
                if w:
                    w.deleteLater()
        self.results_section.hide()

    def _analyze_emotion(self):
        if not self.current_file:
            return
        if not MODEL_AVAILABLE:
            QMessageBox.critical(self, "Model Not Available", f"Cannot run analysis:\n{IMPORT_ERROR}")
            return

        self.analyze_btn.setEnabled(False)
        self.emotion_label.setText("Processing audio...")
        self._set_emotion_svg('neutral')
        self.circular_progress.setValue(0)

        self.worker = EmotionWorker(self.current_file)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, results: dict):
        try:
            self._update_top_display(results)
            self._populate_grid(results)
            self.results_section.show()
        except Exception as e:
            QMessageBox.critical(self, "Display Error", f"Failed to render results:\n{str(e)}")
        finally:
            self.analyze_btn.setEnabled(True)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Analysis Error", msg)
        self.analyze_btn.setEnabled(True)
        self._reset_display()

    def _update_top_display(self, results: dict):
        emotion = results.get('overall_emotion', 'neutral')
        conf = float(results.get('confidence', 0.0))
        self.emotion_label.setText(emotion.upper())
        self.confidence_label.setText(f"Confidence: {conf*100:.1f}%")
        self.circular_progress.setValue(int(conf * 100))

        color_map = {
            'angry': '#FF6B6B', 'calm': '#20B2AA', 'disgust': '#32CD32',
            'fearful': '#9370DB', 'happy': '#FFD700', 'neutral': '#4A90E2',
            'sad': '#6A5ACD', 'surprised': '#FFA500'
        }
        self.circular_progress.setProgressColor(color_map.get(emotion.lower(), '#4A90E2'))
        self._set_emotion_svg(emotion)

    def _populate_grid(self, results: dict):
        # Clear
        for i in reversed(range(self.scores_grid.count())):
            w = self.scores_grid.itemAt(i).widget()
            if w:
                w.deleteLater()

        scores = results.get('all_emotion_scores', {})
        if not scores:
            return

        # Stable ordering: sort by score desc
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        row, col = 0, 0
        for emotion, score in items:
            self.scores_grid.addWidget(self._score_card(emotion, float(score)), row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

    def _score_card(self, emotion: str, score: float) -> QWidget:
        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #F8F9FA; border-radius: 10px; padding: 12px; border: 2px solid #E9ECEF; }")
        card.setFixedWidth(180)
        v = QVBoxLayout(card)
        v.setSpacing(8)

        top = QHBoxLayout()
        face = QSvgWidget()
        face.setFixedSize(28, 28)
        face.load(QByteArray(emotion_svg_xml(emotion).encode('utf-8')))
        top.addWidget(face)
        lbl = QLabel(emotion.upper())
        lbl.setStyleSheet("QLabel { font-weight: bold; color: #1F2937; font-size: 13px; }")
        top.addWidget(lbl)
        top.addStretch(1)
        v.addLayout(top)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(score * 100))
        bar.setTextVisible(False)
        color = {
            'angry': '#FF6B6B', 'calm': '#20B2AA', 'disgust': '#32CD32',
            'fearful': '#9370DB', 'happy': '#FFD700', 'neutral': '#4A90E2',
            'sad': '#6A5ACD', 'surprised': '#FFA500'
        }.get(emotion.lower(), '#4A90E2')
        bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #E5E7EB; border-radius: 6px; background: white; height: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {color}; border-radius: 6px;
            }}
        """)
        v.addWidget(bar)

        pct = QLabel(f"{score*100:.1f}%")
        pct.setAlignment(Qt.AlignRight)
        pct.setStyleSheet("QLabel { color: #6B7280; font-size: 12px; font-weight: 600; }")
        v.addWidget(pct)

        return card

    # ---------- Window lifecycle ----------
    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
        finally:
            event.accept()

# ================== MAIN ==================
def main():
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        win = SERGUI()
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application crashed: {e}")
        print(traceback.format_exc())
        try:
            QMessageBox.critical(None, "Application Error", f"The application crashed:\n{str(e)}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
