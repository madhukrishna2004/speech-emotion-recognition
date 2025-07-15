import sys
import time
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import threading

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ser.models.emotion_model import predict_emotions
from ser.transcript.transcript_extractor import extract_transcript


SAMPLE_RATE = 22050
CHUNK_DURATION = 2  # seconds
ROLLING_HISTORY = 30  # number of chunks to show on graph


class SERApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_recording = False
        self.emotion_history = []

    def init_ui(self):
        self.setWindowTitle("🎙️ Real-Time Speech Emotion Recognition")
        self.setGeometry(200, 200, 800, 600)

        self.start_btn = QPushButton("Start Recording", self)
        self.start_btn.clicked.connect(self.toggle_recording)

        self.transcript_box = QTextEdit(self)
        self.transcript_box.setReadOnly(True)

        self.emotion_label = QLabel("Detected Emotion: ", self)
        self.emotion_label.setStyleSheet("font-size: 18px;")

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.transcript_box)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio_chunk)

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.start_btn.setText("Stop Recording")
            self.timer.start(CHUNK_DURATION * 1000)
        else:
            self.start_btn.setText("Start Recording")
            self.timer.stop()

    def process_audio_chunk(self):
        audio = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio_data = audio.flatten()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
            audio_path = temp_wav.name

        threading.Thread(target=self.analyze_audio, args=(audio_path,)).start()

    def analyze_audio(self, audio_path):
        try:
            emotions = predict_emotions(audio_path)
            transcript = extract_transcript(audio_path)

            if emotions:
                emotion = emotions[-1][0]
                self.update_emotion_display(emotion)

                self.emotion_history.append(emotion)
                if len(self.emotion_history) > ROLLING_HISTORY:
                    self.emotion_history.pop(0)

                self.update_emotion_plot()

                if emotion in ["angry", "fearful"]:
                    QApplication.beep()  # 🔔 Alert

            if transcript:
                text = " ".join([word for word, _, _ in transcript])
                self.transcript_box.append(f"> {text}")

        except Exception as e:
            self.transcript_box.append(f"[ERROR] {e}")

        finally:
            os.unlink(audio_path)

    def update_emotion_display(self, emotion):
        color = {
            "happy": "green",
            "sad": "blue",
            "angry": "red",
            "fearful": "orange",
            "neutral": "gray",
            "calm": "teal",
            "disgust": "purple",
            "surprised": "magenta"
        }.get(emotion, "black")
        self.emotion_label.setText(f"Detected Emotion: {emotion}")
        self.emotion_label.setStyleSheet(f"color: {color}; font-size: 18px;")

    def update_emotion_plot(self):
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1

        self.ax.clear()
        self.ax.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
        self.ax.set_title("Emotion History (Last 30)")
        self.ax.set_ylabel("Count")
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SERApp()
    gui.show()
    sys.exit(app.exec_())
