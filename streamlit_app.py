import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
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
# DETECTOR FROM SCRATCH (BASE SIMPLE PARA ESCALAR)
# =====================================================
class SimpleDoGDetector:
    """
    Detector base inspirado en DoG.
    - NO usa cv2 para DoG
    - Usa NumPy + SciPy
    - Ideal como base académica para evolucionar a SIFT completo
    """

    def __init__(self, sigma1=1.0, sigma2=2.0, threshold=0.03):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.threshold = threshold

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
            min_distance=5,
            threshold_abs=self.threshold,
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
# VISUALIZATION
# =====================================================
def draw_keypoints(image, keypoints):
    output = image.copy()

    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for y, x in keypoints:
        cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), 1)

    return output


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("Implementación From Scratch + Live Demo")
st.subheader("SIFT inspirado en DoG + Webcam en tiempo real")

st.sidebar.header("Configuración")
mode = st.sidebar.radio(
    "Selecciona modo:",
    [
        "Evaluación de imágenes",
        "Live Demo Webcam",
    ],
)

sigma1 = st.sidebar.slider("Sigma 1", 0.5, 5.0, 1.0, 0.1)
sigma2 = st.sidebar.slider("Sigma 2", 0.5, 6.0, 2.0, 0.1)
threshold = st.sidebar.slider("Threshold", 0.001, 0.1, 0.03, 0.001)


detector = SimpleDoGDetector(
    sigma1=sigma1,
    sigma2=sigma2,
    threshold=threshold,
)


# =====================================================
# MODO 1: EVALUACIÓN DE IMÁGENES
# =====================================================
if mode == "Evaluación de imágenes":
    st.header("Evaluación con imágenes")

    uploaded_files = st.file_uploader(
        "Sube imágenes (Lenna, Baboon, Cameraman, bebé, anciano, perro, gato, etc.)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    results = []

    if uploaded_files:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.warning(f"No se pudo leer: {file.name}")
                continue

            result = detector.process(image)
            vis = draw_keypoints(image, result["keypoints"])

            st.subheader(file.name)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original")
            with col2:
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Keypoints")

            st.write(f"Tiempo de procesamiento: {result['time']:.4f} s")
            st.write(f"Número de keypoints: {result['count']}")

            results.append(
                {
                    "Imagen": file.name,
                    "Tiempo (s)": round(result["time"], 4),
                    "Keypoints": result["count"],
                }
            )

    if results:
        df = pd.DataFrame(results)

        st.header("Tabla resumen")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar resultados CSV",
            data=csv,
            file_name="resultados_sift.csv",
            mime="text/csv",
        )

        st.header("Gráficas")

        fig1, ax1 = plt.subplots()
        ax1.bar(df["Imagen"], df["Tiempo (s)"])
        ax1.set_title("Tiempo de procesamiento por imagen")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.bar(df["Imagen"], df["Keypoints"])
        ax2.set_title("Número de keypoints detectados")
        plt.xticks(rotation=45)
        st.pyplot(fig2)


# =====================================================
# MODO 2: LIVE DEMO WEBCAM (streamlit-webrtc)
# =====================================================

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = SimpleDoGDetector(
            sigma1=sigma1,
            sigma2=sigma2,
            threshold=threshold,
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = self.detector.process(img)
        vis = draw_keypoints(img, result["keypoints"])

        cv2.putText(
            vis,
            f"KP: {result['count']} | Time: {result['time']:.3f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(vis, format="bgr24")


if mode == "Live Demo Webcam":
    st.header("Live Demo Webcam")
    st.write("Captura desde cámara del navegador")

    picture = st.camera_input("Toma una foto")

    if picture is not None:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        result = detector.process(frame)
        vis = draw_keypoints(frame, result["keypoints"])

        st.image(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
            caption=f"KP: {result['count']} | Time: {result['time']:.4f}s\"
        )
