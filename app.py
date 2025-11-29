import streamlit as st
import tempfile
import os
from predict import ShopliftingPrediction, CameraPrediction

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🎥",
    layout="centered"
)

# ---------------- HEADER ----------------
st.title("🎥 Smart Surveillance System")
st.markdown(
    """
    Our **Smart Surveillance System** detects suspicious activities in real-time using AI-enhanced video and audio analysis.
    
    ### 🔍 Why it matters
    - Prevents theft, vandalism, and suspicious behavior.
    - Provides **instant alerts** for unusual activities.
    - Enhances video quality even in **low-light** conditions.

    Upload a video or use your **live camera** to analyze and detect anomalies in real-time.
    """
)

# ---------------- SIDEBAR INFO ----------------
st.sidebar.header("🧠 Problem & Solution")
st.sidebar.markdown(
    """
    **Problem:** Traditional surveillance systems miss subtle suspicious behaviors or rely heavily on human monitoring.  
    **Solution:** This system uses **AI-powered video and audio analysis** for accurate, automated detection.  

    **Benefits:**  
    - Detect thefts and abnormal sounds automatically.  
    - Reduce manual monitoring.  
    - Get enhanced video output with real-time alerting.
    """
)

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "📂 Upload a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

# ---------------- MODEL PATHS ----------------
MODEL_PATH = r"C:\Users\rushi\OneDrive\Desktop\my_project\lrcn_model.keras"
YAMNET_PATH = r"C:\Users\rushi\OneDrive\Desktop\my_project\yamnet-tensorflow2-yamnet-v1"
CLASS_MAP_PATH = r"C:\Users\rushi\OneDrive\Desktop\my_project\yamnet-tensorflow2-yamnet-v1\assets\yamnet_class_map.csv"
ENLIGHTEN_MODEL_PATH = r"C:\Users\rushi\OneDrive\Desktop\my_project\EnlightenGAN-inference\enlighten_inference\enlighten.onnx"
BEEP_AUDIO_PATH = r"C:\Users\rushi\OneDrive\Desktop\my_project\beep-125033.mp3"

FRAME_WIDTH = 90
FRAME_HEIGHT = 90
SEQUENCE_LENGTH = 160

# ---------------- SESSION STATE ----------------
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None

# ---------------- CREATE PREDICTOR INSTANCE ----------------
st.sidebar.subheader("⚙️ Model Initialization")
with st.spinner("Loading AI models... please wait ⏳"):
    try:
        predictor = ShopliftingPrediction(
            model_path=MODEL_PATH,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            sequence_length=SEQUENCE_LENGTH,
            yamnet_path=YAMNET_PATH,
            class_map_path=CLASS_MAP_PATH,
            enlighten_model_path=ENLIGHTEN_MODEL_PATH,  # ✅ Enable enhancement
            beep_path=BEEP_AUDIO_PATH
        )
        predictor.load_model()
        st.sidebar.success("✅ Models loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# ---------------- VIDEO FILE PROCESSING ----------------
if uploaded_file:
    st.subheader("📥 Uploaded Video")
    st.video(uploaded_file, start_time=0, format="video/mp4", width=480)

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input_path = temp_input.name
    temp_input.close()

    output_path = os.path.join(os.getcwd(), "output_video.mp4")

    if st.button("🚀 Run Anomaly Detection"):
        with st.spinner("🔧 Processing video... Please wait ⏳"):
            try:
                st.info("🚀 Detecting suspicious activity in uploaded video...")
                predictor.Predict_Video(temp_input_path, output_path, live_preview=False)
                st.session_state.output_video_path = output_path
                st.success("✅ Processing completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

# ---------------- DISPLAY OUTPUT VIDEO ----------------
if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
    st.subheader("📤 Annotated Output Video")
    with open(st.session_state.output_video_path, "rb") as f:
        video_bytes = f.read()
        st.video(video_bytes, start_time=0, format="video/mp4", width=480)

    st.download_button(
        label="📥 Download Annotated Video",
        data=video_bytes,
        file_name="output_video.mp4",
        mime="video/mp4"
    )



# ---------------- LIVE CAMERA DETECTION ----------------
st.markdown("---")
st.subheader("🎥 Live Camera Detection")

if st.button("▶️ Start Live Detection"):
    st.info("Press **'q'** in the preview window to stop detection.")
    with st.spinner("Starting camera..."):
        try:
            live_detector = CameraPrediction(predictor)  # ✅ integrated class
            live_detector.start_camera_detection()
            st.success("✅ Live detection ended successfully.")
        except Exception as e:
            st.error(f"❌ Error starting live detection: {str(e)}")
