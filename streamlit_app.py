import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="SIFT From Scratch + Live Demo",
    layout="wide"
)


# =====================================================
# DETECTOR OPTIMIZADO (DoG simplificado)
# =====================================================
class SimpleDoGDetector:
    """
    Detector inspirado en SIFT usando Difference of Gaussians.

    Reglas:
    - NO usa cv2 para DoG
    - Usa NumPy + SciPy
    - Optimizado para Streamlit Live Demo
    - OpenCV solo para captura / dibujo / conversión
    """

    def __init__(
        self,
        sigma1=1.0,
        sigma2=2.0,
        threshold=0.08,
        min_distance=12,
        max_keypoints=150,
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
# DIBUJAR KEYPOINTS
# =====================================================
def draw_keypoints(image, keypoints):
    output = image.copy()

    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    for y, x in keypoints:
        cv2.circle(
            output,
            (int(x), int(y)),
            3,
            (0, 255, 0),
            1,
        )

    return output


# =====================================================
# TITULO PRINCIPAL
# =====================================================
st.title("Implementación From Scratch + Live Demo")
st.subheader("SIFT inspirado en DoG + Webcam en tiempo real")


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Configuración")

mode = st.sidebar.radio(
    "Selecciona modo:",
    [
        "Evaluación de imágenes",
        "Live Demo Webcam",
    ],
)

sigma1 = st.sidebar.slider(
    "Sigma 1",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

sigma2 = st.sidebar.slider(
    "Sigma 2",
    min_value=0.5,
    max_value=6.0,
    value=2.0,
    step=0.1,
)

threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.001,
    max_value=0.1,
    value=0.08,
    step=0.001,
)


# =====================================================
# DETECTOR GLOBAL
# =====================================================
detector = SimpleDoGDetector(
    sigma1=sigma1,
    sigma2=sigma2,
    threshold=threshold,
)


# =====================================================
# MODO 1 — EVALUACIÓN DE IMÁGENES
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
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Imagen original",
                )

            with col2:
                st.image(
                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                    caption="Keypoints detectados",
                )

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
            label="Descargar CSV",
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
# CLASE PARA WEBCAM EN VIVO
# =====================================================
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

        # Reducir resolución para estabilidad
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
# MODO 2 — LIVE DEMO WEBCAM
# =====================================================
if mode == "Live Demo Webcam":
    st.header("Live Demo Webcam")
    st.write("Detección de keypoints en tiempo real")

    st.info(
        "Versión optimizada para estabilidad en Streamlit Cloud."
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
