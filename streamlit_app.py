import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from PIL import Image


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="SIFT From Scratch + Live Demo", layout="wide")

# =====================================================
# DETECTOR OPTIMIZADO PARA LIVE DEMO (FAST VERSION)
# =====================================================

class SimpleDoGDetector:
    """
    Versión optimizada para Streamlit Live Demo.
    Prioriza fluidez sobre precisión extrema.
    """

    def __init__(
        self,
        sigma1=1.0,
        sigma2=2.0,
        threshold=0.08,   # más agresivo
        min_distance=12,  # menos puntos cercanos
        max_keypoints=150 # límite de keypoints
    ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold = threshold
        self.min_distance = min_distance
        self.max_keypoints = max_keypoints

    def preprocess(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32) / 255.0
        return gray

    def compute_dog(self, gray):
        g1 = gaussian_filter(gray, sigma=self.sigma1)
        g2 = gaussian_filter(gray, sigma=self.sigma2)
        dog = g2 - g1
        return dog

    def detect_keypoints(self, dog):
        coords = peak_local_max(
            np.abs(dog),
            min_distance=self.min_distance,
            threshold_abs=self.threshold,
            num_peaks=self.max_keypoints,
            exclude_border=False,
        )
        return coords

    def process(self, image):
        start = time.perf_counter()

        gray = self.preprocess(image)
        dog = self.compute_dog(gray)
        keypoints = self.detect_keypoints(dog)

        elapsed = time.perf_counter() - start

        return {
            "gray": gray,
            "dog": dog,
            "keypoints": keypoints,
            "time": elapsed,
            "count": len(keypoints),
        }


# =====================================================
# LIVE DEMO WEBCAM OPTIMIZADO (streamlit-webrtc)
# =====================================================

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = SimpleDoGDetector(
            sigma1=1.0,
            sigma2=2.0,
            threshold=0.08,
            min_distance=12,
            max_keypoints=150,
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # =================================================
        # REDUCCIÓN DE RESOLUCIÓN → MUCHÍSIMA MEJORA
        # =================================================
        img = cv2.resize(img, (480, 360))

        result = self.detector.process(img)
        vis = draw_keypoints(img, result["keypoints"])

        cv2.putText(
            vis,
            f"KP: {result['count']} | Time: {result['time']:.3f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(
            vis,
            format="bgr24"
        )


# =====================================================
# MODO 2: LIVE DEMO WEBCAM
# =====================================================

if mode == "Live Demo Webcam":
    st.header("Live Demo Webcam")
    st.write("Detección de keypoints en tiempo real (versión optimizada)")

    st.info(
        "Versión optimizada para estabilidad en Streamlit Cloud "
        "y procesamiento en tiempo real."
    )

    webrtc_streamer(
        key="live-demo",

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        },

        async_processing=True,
    )
        async_processing=True,
    )
