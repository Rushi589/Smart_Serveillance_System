Smart Surveillance System (Audio + Video Anomaly Detection)

The Smart Surveillance System is an AI-powered multimodal monitoring solution that detects suspicious activities in real time using both video and audio inputs. Traditional CCTV systems struggle with low-light conditions, noisy environments, and manual monitoring limitations. This project integrates deep learning, audio-video enhancement, and multimodal fusion to provide intelligent and automated surveillance alerts.

📌 Key Features

1.Multimodal Detection (Video + Audio)
Combines insights from both video frames and environmental audio to improve anomaly detection accuracy.

2.Low-Light Video Enhancement (EnlightenGAN)
Enhances visibility in dark environments, making nighttime surveillance more reliable.

3.Audio Anomaly Detection (Fine-Tuned YAMNet)
Detects events like shouting, glass breaking, impact sounds, and distress cues.

4.Video Anomaly Detection (LRCN Model)
Identifies actions such as fighting, robbery, suspicious movement, and abnormal behaviors.

5.Spectral Gating for Noise Reduction
Reduces background noise to improve audio classification accuracy.

6.Real-Time Alerts
Triggers an alarm/beep when anomalies are detected.

🛠️ Tools & Technologies Used

#Video Processing

TensorFlow / Keras
LRCN (CNN + LSTM)
EnlightenGAN for low-light enhancement
OpenCV
MoviePy
NumPy / Pandas

#Audio Processing

YAMNet (fine-tuned)
TensorFlow Hub
Librosa for audio preprocessing
Spectral Gating for denoising


📚 Datasets Used

1. Audio Dataset – MIVIA Audio Events Dataset
https://mivia.unisa.it/datasets-request/

2. Video Dataset – UCF Crime Dataset
https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
