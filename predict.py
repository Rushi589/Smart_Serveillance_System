import os
import sys
import cv2
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import librosa
import threading
from collections import deque
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
import onnxruntime as ort

# Attempt to import sounddevice for live microphone capture (optional)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True

    # Force correct device index & samplerate to avoid _sounddevice_data / PortAudio errors
    def get_working_input_device():
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                return idx
        return None

    WORKING_MIC_INDEX = get_working_input_device()
    if WORKING_MIC_INDEX is None:
        SOUNDDEVICE_AVAILABLE = False
        print("⚠️ No working microphone found.")

except Exception as e:
    SOUNDDEVICE_AVAILABLE = False
    WORKING_MIC_INDEX = None
    print(f"⚠️ sounddevice import failed: {e}")


# ------------------ EnlightenGAN ONNX MODEL ------------------
class EnlightenOnnxModel:
    def __init__(self, model_path):
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
            print("✅ Using CUDAExecutionProvider (GPU)")
        else:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print("⚠️ CUDA not available, using CPUExecutionProvider")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))
        arr = img_rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, 0)
        return arr

    def postprocess(self, arr, orig_shape):
        arr = np.squeeze(arr)
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        arr = cv2.resize(arr, (orig_shape[1], orig_shape[0]))
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = cv2.convertScaleAbs(arr, alpha=1.2, beta=10)
        return arr

    def predict(self, frame):
        inp = self.preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: inp})[0]
        return self.postprocess(output, frame.shape)

    @staticmethod
    def enhance_video_with_enlighten(input_video_path, model_path):
        clip = VideoFileClip(input_video_path)
        model = EnlightenOnnxModel(model_path)
        enhanced_frames = []
        frame_count = 0

        for frame in clip.iter_frames():
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                enhanced_bgr = model.predict(frame_bgr)
            except Exception as e:
                print(f"⚠️ Error at frame {frame_count}: {e}")
                enhanced_bgr = frame_bgr

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"✅ Enhanced {frame_count} frames...")

            enhanced_frames.append(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))

        clip.close()
        return enhanced_frames

    @staticmethod
    def extract_audio_from_video(video_path, sr=16000):
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            print("⚠️ No audio found in video")
            return None, None

        audio_path = "temp_audio.wav"
        clip.audio.write_audiofile(audio_path, fps=sr)
        waveform, sr = librosa.load(audio_path, sr=sr)
        clip.close()

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return waveform, sr


# ------------------ SHOPLIFTING PREDICTION (file-mode) ------------------
class ShopliftingPrediction:
    """
    This class preserves your original file-based pipeline. It loads the LRCN model,
    YAMNet (if available), runs detection over a video file (optionally enhanced by Enlighten),
    and writes annotated output with optional beep overlays.
    """

    def __init__(self, model_path, frame_width, frame_height, sequence_length,
                 yamnet_path, class_map_path, enlighten_model_path=None, beep_path=None):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.video_message = "Ready"
        self.audio_message = "Audio Ready"
        self.yamnet_path = yamnet_path
        self.class_map_path = class_map_path
        self.enlighten_model_path = enlighten_model_path
        self.alarm_triggered = False
        self.beep_path = beep_path  # path to beep mp3/wav

        # Load YAMNet + classes
        if self.yamnet_path and os.path.exists(self.yamnet_path):
            try:
                self.yamnet_model = tf.saved_model.load(self.yamnet_path)
            except Exception as e:
                print("⚠️ Could not load YAMNet model:", e)
                self.yamnet_model = None
        else:
            self.yamnet_model = None

        if self.class_map_path and os.path.exists(self.class_map_path):
            try:
                self.class_names = pd.read_csv(self.class_map_path)["display_name"].tolist()
            except Exception as e:
                print("⚠️ Could not load class map:", e)
                self.class_names = []
        else:
            self.class_names = []

        self.suspicious_classes = {
            "Shout", "Yell", "Screaming", "Bellow", "Whoop", "Crying, sobbing", "Whimper", "Wail, moan",
            "Whispering", "Chatter", "Children shouting", "Glass", "Shatter", "Smash, crash", "Chink, clink",
            "Wood", "Crack", "Splinter", "Chop", "Bang", "Slap, smack", "Whack, thwack", "Explosion",
            "Fireworks", "Firecracker", "Gunshot, gunfire", "Machine gun", "Artillery fire", "Door", "Slam",
            "Squeak", "Knock", "Cupboard open or close", "Drawer open or close", "Keys jangling", "Walk, footsteps",
            "Run", "Shuffle", "Car alarm", "Siren", "Alarm clock", "Buzzer", "Smoke detector, smoke alarm",
            "Fire alarm", "Hammer", "Drill", "Jackhammer", "Sawing", "Filing (rasp)", "Sanding",
            "Power tool", "Dog", "Bark", "Growling", "Howl", "Whimper (dog)"
        }

        # --- NEW: load fine-tuned classifier and use custom suspicious class set ---
        # Minimal, safe additions:
        self.fine_tuned_classifier_path = r"C:\Users\rushi\OneDrive\Desktop\my_project\fine_tuned_yamnet_classifier.h5"
        self.fine_tuned_available = False
        # set our target suspicious classes exactly as requested
        self.class_names = ['glass', 'gunshots', 'screams']
        # update suspicious set to only include these labels (for file-mode textual checks)
        self.suspicious_classes = set(self.class_names)

        # Try to load the fine-tuned classifier and ensure we have a yamnet function for embeddings
        try:
            if os.path.exists(self.fine_tuned_classifier_path):
                # load classifier (expects embeddings as input)
                self.classifier = tf.keras.models.load_model(self.fine_tuned_classifier_path)
                # ensure we have a yamnet model to extract embeddings (use existing yamnet_path if loaded, else try default)
                if self.yamnet_model is None:
                    # attempt to load default YAMNet if present beside classifier (best-effort, silent if fails)
                    try:
                        default_yamnet_path = r"C:\Users\rushi\OneDrive\Desktop\yamnet\yamnet-tensorflow2-yamnet-v1"
                        if os.path.exists(default_yamnet_path):
                            self.yamnet_model = tf.saved_model.load(default_yamnet_path)
                    except Exception:
                        pass

                if self.yamnet_model is not None:
                    # get callable signature
                    try:
                        self.yamnet_func = self.yamnet_model.signatures['serving_default']
                    except Exception:
                        self.yamnet_func = None
                    self.fine_tuned_available = True
                    print("✅ Fine-tuned classifier loaded and ready.")
                else:
                    # classifier loaded but yamnet embeddings unavailable (we keep classifier but flag)
                    self.yamnet_func = None
                    self.fine_tuned_available = False
                    print("⚠️ Fine-tuned classifier loaded but YAMNet embeddings not available.")
            else:
                self.classifier = None
                self.yamnet_func = None
                print("⚠️ Fine-tuned classifier file not found at expected path.")
        except Exception as e:
            print("⚠️ Error loading fine-tuned classifier:", e)
            self.classifier = None
            self.yamnet_func = None
            self.fine_tuned_available = False

    def load_model(self):
        # Keep behavior identical to your original code
        self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)

    def generate_message_content(self, probability, label):
        if label == 0:
            if probability <= 75:
                self.video_message = "There is little chance of theft"
            elif probability <= 85:
                self.video_message = "High probability of theft"
            else:
                self.video_message = "Very high probability of theft"
        elif label == 1:
            if probability <= 75:
                self.video_message = "The movement is confusing, watch"
            elif probability <= 85:
                self.video_message = "I think it's normal, but it's better to watch"
            else:
                self.video_message = "Movement is normal"

    def Pre_Process_Video(self, current_frame, previous_frame):
        diff = cv2.absdiff(current_frame, previous_frame)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        resized_frame = cv2.resize(diff, (self.frame_height, self.frame_width))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0
        return normalized_frame

    def draw_text(self, img, text, pos, font, scale, color, thickness, max_width):
        words = text.split(" ")
        line = ""
        y = pos[1]
        for word in words:
            test_line = line + word + " "
            (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
            if w > max_width:
                cv2.putText(img, line, (pos[0], y), font, scale, color, thickness, cv2.LINE_AA)
                line = word + " "
                y += int(h * 1.5)
            else:
                line = test_line
        if line:
            cv2.putText(img, line, (pos[0], y), font, scale, color, thickness, cv2.LINE_AA)

    # Continuous beep thread
    def start_beeping(self):
        def _beep_loop():
            while self.alarm_triggered:
                try:
                    if sys.platform.startswith("win"):
                        import winsound
                        winsound.Beep(1000, 500)
                    elif sys.platform == "darwin":
                        os.system("afplay /System/Library/Sounds/Glass.aiff")
                    else:
                        os.system("play -nq -t alsa synth 0.3 sine 1000")
                except Exception as e:
                    print("Beep error:", e)
        threading.Thread(target=_beep_loop, daemon=True).start()

    def predict_audio_chunk(self, waveform):
        """
        waveform: 1D numpy arr (samples) at sample rate expected by yamnet (e.g. 16000)
        Keeps same behavior as your original code, but prefer the fine-tuned classifier if available.
        """
        # If fine-tuned classifier and yamnet function are available, use them
        if getattr(self, "classifier", None) is not None and getattr(self, "yamnet_func", None) is not None:
            try:
                # ensure waveform is numpy float32
                waveform = np.array(waveform, dtype=np.float32)
                # normalize
                wave_max = np.max(np.abs(waveform)) + 1e-9
                waveform = waveform / wave_max

                # convert to tf and call yamnet to get embeddings
                waveform_tf = tf.constant(waveform, dtype=tf.float32)
                waveform_tf = tf.reshape(waveform_tf, [-1])


                yamnet_out = self.yamnet_func(waveform=waveform_tf)
                # Try known embedding keys in order
                possible_keys = ['output_1', 'embedding', 'embeddings', 'output_0']
                embeddings = None
                for k in possible_keys:
                    if k in yamnet_out:
                        embeddings = yamnet_out[k]
                        break
                # Fallback: take first tensor if keys differ
                if embeddings is None:
                    embeddings = list(yamnet_out.values())[0]


                mean_embed = tf.reduce_mean(embeddings, axis=0).numpy()
                preds = self.classifier.predict(np.expand_dims(mean_embed, axis=0), verbose=0)[0]
                pred_idx = int(np.argmax(preds))
                confidence = float(np.max(preds))
                # ensure index-safe mapping
                if pred_idx < 0 or pred_idx >= len(self.class_names):
                    pred_class = "Unknown"
                else:
                    pred_class = self.class_names[pred_idx]

                # set audio_message depending on requested suspicious classes only
                if pred_class in self.suspicious_classes and confidence >= 0.0:
                    # Use a reasonably high threshold for reporting suspicious (you can tune)
                    if confidence >= 0.5:
                        self.audio_message = f"Suspicious Audio: {pred_class} ({confidence:.2f})"
                    else:
                        # below strict suspicious threshold but still one of target classes
                        self.audio_message = f"Suspicious Audio (low conf): {pred_class} ({confidence:.2f})"
                else:
                    self.audio_message = f"Normal Audio ({confidence:.2f})"

                return confidence, pred_class

            except Exception as e:
                print("⚠️ Fine-tuned YAMNet prediction error:", e)
                # fallback to original yamnet behavior below

        # Fallback: original YAMNet scoring (keeps original behavior)
        if self.yamnet_model is None or not self.class_names:
            return 0.0, "Unknown"
        try:
            scores, _, _ = self.yamnet_model(waveform)
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            best_idx = np.argmax(mean_scores)
            best_class = self.class_names[best_idx] if best_idx < len(self.class_names) else "Unknown"
            best_score = mean_scores[best_idx]
            # treat only names from self.suspicious_classes as suspicious
            if best_class in self.suspicious_classes:
                self.audio_message = f"Suspicious Audio: {best_class} ({best_score:.2f})"
            else:
                self.audio_message = f"Normal Audio: {best_class} ({best_score:.2f})"
            return float(best_score), best_class
        except Exception as e:
            print("⚠️ YAMNet error:", e)
            return 0.0, "Unknown"

    # Helper to loop beep audio
    def loop_audio(self, clip, duration):
        loops = math.ceil(duration / clip.duration)
        concatenated = concatenate_audioclips([clip] * loops)
        return concatenated.subclip(0, duration)

    def _merge_consecutive_indices_into_segments(self, indices, fps):
        if not indices:
            return []
        indices = sorted(indices)
        segments = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                segments.append((start / fps, (prev + 1) / fps))
                start = prev = idx
        segments.append((start / fps, (prev + 1) / fps))
        return segments

    def Predict_Video(self, video_file_path, output_file_path, live_preview=False):
        """
        Keep this method fully intact (file-based). Only small guards added for robustness.
        """
        waveform, sr = EnlightenOnnxModel.extract_audio_from_video(video_file_path, sr=16000)
        audio_ptr = 0 if waveform is not None else None

        if self.enlighten_model_path:
            print("✨ Enhancing video frames using EnlightenGAN...")
            enhanced_frames = EnlightenOnnxModel.enhance_video_with_enlighten(video_file_path, self.enlighten_model_path)
        else:
            clip_tmp = VideoFileClip(video_file_path)
            enhanced_frames = [frame for frame in clip_tmp.iter_frames()]
            clip_tmp.close()

        clip = VideoFileClip(video_file_path)
        fps = clip.fps
        print(f"🔎 Detected FPS: {fps}")

        previous_frame = None
        frames_queue = []
        video_prob = 0
        strip_height = 280
        max_width = int(clip.w) - 20
        output_frames = []
        red_frame_indices = []

        # initialize label holder for file-mode decision
        last_video_label = None

        for idx, frame_rgb in enumerate(enhanced_frames):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if previous_frame is None:
                previous_frame = frame_bgr
            normalized_frame = self.Pre_Process_Video(frame_bgr, previous_frame)
            previous_frame = frame_bgr.copy()
            frames_queue.append(normalized_frame)

            if len(frames_queue) == self.sequence_length:
                probabilities = self.model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(probabilities)
                probability = math.floor(np.max(probabilities) * 100)
                # store last_video_label so the alert decision can reference it
                last_video_label = int(predicted_label)
                self.generate_message_content(probability, predicted_label)
                self.video_message = f"{self.video_message}: {probability}%"
                video_prob = probability
                frames_queue = []

            if audio_ptr is not None and waveform is not None:
                samples_per_frame = int(16000 / fps)
                if audio_ptr + samples_per_frame < len(waveform):
                    chunk = waveform[audio_ptr:audio_ptr + samples_per_frame]
                    _, _ = self.predict_audio_chunk(chunk)
                    audio_ptr += samples_per_frame

            audio_suspicious = "Suspicious" in self.audio_message
            is_red = False
            # NEW: require label == 0 for video-prob-based red alerts
            if (video_prob > 80 and last_video_label == 0) or audio_suspicious:
                alarm_message = "Signal is RED 🚨"
                alarm_color = (0, 0, 255)
                is_red = True
            elif video_prob > 60:
                alarm_message = "Signal is YELLOW ⚠️"
                alarm_color = (0, 255, 255)
            else:
                alarm_message = "Signal is GREEN ✅"
                alarm_color = (0, 200, 0)

            if is_red:
                red_frame_indices.append(idx)
                if not self.alarm_triggered and live_preview:
                    self.alarm_triggered = True
                    self.start_beeping()
            else:
                if self.alarm_triggered and live_preview:
                    self.alarm_triggered = False

            strip = np.ones((strip_height, frame_bgr.shape[1], 3), dtype=np.uint8) * 255
            self.draw_text(strip, f"Video Log: {self.video_message}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, max_width)
            self.draw_text(strip, f"Audio Log: {self.audio_message}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, max_width)
            self.draw_text(strip, f"System Status: {alarm_message}", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, alarm_color, 3, max_width)

            combined = np.vstack((frame_bgr, strip))
            if live_preview:
                cv2.imshow("Shoplifting Monitor", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            output_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        final_clip = ImageSequenceClip(output_frames, fps=fps)

        original_audio = clip.audio
        beep_segments = []

        if red_frame_indices and self.beep_path and os.path.exists(self.beep_path):
            segments = self._merge_consecutive_indices_into_segments(red_frame_indices, fps)
            try:
                beep_audio = AudioFileClip(self.beep_path)
                for start_t, end_t in segments:
                    seg_dur = max(0.0, end_t - start_t)
                    if seg_dur <= 0:
                        continue
                    beep_seg = self.loop_audio(beep_audio, seg_dur)
                    beep_seg = beep_seg.set_start(start_t)
                    beep_segments.append(beep_seg)
            except Exception as e:
                print(f"⚠️ Could not load or process beep audio: {e}")
                beep_segments = []

        if original_audio is not None and beep_segments:
            final_clip = final_clip.set_audio(CompositeAudioClip([original_audio] + beep_segments))
        elif original_audio is not None:
            final_clip = final_clip.set_audio(original_audio)
        elif beep_segments:
            final_clip = final_clip.set_audio(CompositeAudioClip(beep_segments))
        else:
            final_clip = final_clip.set_audio(None)

        final_clip.write_videofile(output_file_path, fps=fps, codec="libx264", audio_codec="aac")

        clip.close()
        final_clip.close()
        cv2.destroyAllWindows()
        self.alarm_triggered = False
        print(f"🎉 Prediction video saved: {output_file_path}")


# ------------------ CAMERA (real-time) PREDICTION ------------------
class CameraPrediction:
    """
    CameraPrediction is a real-time camera mode that:
    - Captures frames from webcam (OpenCV)
    - Buffers frames into sequences of length `sequence_length` and runs the same model
    - Captures microphone audio (optional, via sounddevice) and runs YAMNet over audio chunks
    - Displays live annotated window and triggers alarms (beeping) like file mode
    """

    def __init__(self, shop_predictor: ShopliftingPrediction, device_index=0,
                 audio_samplerate=16000, audio_chunk_seconds=0.5):
        """
        shop_predictor: an instance of ShopliftingPrediction with model loaded (model & yamnet)
        device_index: camera index for cv2.VideoCapture
        audio_samplerate: desired sample rate for audio capture
        audio_chunk_seconds: size of chunk used to run audio detection (seconds)
        """
        self.predictor = shop_predictor
        if self.predictor.model is None:
            raise ValueError("Predictor.model is not loaded. Call shop_predictor.load_model() before using CameraPrediction.")

        self.device_index = device_index
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device_index}")

        self.audio_samplerate = audio_samplerate
        self.audio_chunk_seconds = audio_chunk_seconds
        self.audio_chunk_samples = int(self.audio_samplerate * self.audio_chunk_seconds)

        # frame buffer for video sequence
        self.frames_buffer = deque(maxlen=self.predictor.sequence_length)
        self.previous_frame = None

        # audio buffering
        self.audio_buffer = deque()
        self.audio_lock = threading.Lock()
        self.audio_stream = None
        self.use_audio = SOUNDDEVICE_AVAILABLE and (self.predictor.yamnet_model is not None)

        self.alarm_triggered = False

        # thread flags
        self._stop_flag = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice.InputStream callback: append captured audio to internal buffer.
        indata shape: (frames, channels). We'll convert to mono if needed.
        """
        if status:
            # optionally print status messages
            pass
        try:
            if indata.ndim > 1:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata
            with self.audio_lock:
                self.audio_buffer.extend(mono.tolist())
        except Exception as e:
            print("Audio callback error:", e)

    def _start_audio_stream(self):
      """Start microphone stream and buffer audio."""
      if not SOUNDDEVICE_AVAILABLE:
        print("⚠️ sounddevice not available — audio detection disabled.")
        return
      try:
        self.audio_samplerate = 48000  # Use actual mic sample rate
        self.audio_chunk_seconds = 1.0  # 1-second chunks
        self.audio_chunk_samples = int(self.audio_samplerate * self.audio_chunk_seconds)

        self.audio_stream = sd.InputStream(
            device=WORKING_MIC_INDEX if WORKING_MIC_INDEX is not None else None,
            samplerate=self.audio_samplerate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        self.audio_stream.start()
        print("🎤 Audio stream started.")
      except Exception as e:
        print("⚠️ Could not start audio stream:", e)
        self.audio_stream = None

    def _stop_audio_stream(self):
        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
                print("🎤 Audio stream stopped.")
        except Exception as e:
            print("⚠️ Error stopping audio stream:", e)

    def _get_audio_chunk_if_available(self):
      """Return 1-second audio chunk for YAMNet, resampled to 16 kHz."""
      with self.audio_lock:
        if len(self.audio_buffer) >= self.audio_chunk_samples:
            samples = [self.audio_buffer.popleft() for _ in range(self.audio_chunk_samples)]
            audio_chunk = np.array(samples, dtype=np.float32)

            # Debug: check raw amplitude
            rms = np.sqrt(np.mean(audio_chunk**2))
            peak = np.max(np.abs(audio_chunk))
            print(f"🔊 Raw audio RMS: {rms:.4f}, peak: {peak:.4f}")

            # Amplify quiet signals if RMS < 0.05
            if rms < 0.05:
                audio_chunk = audio_chunk * 10.0
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                print("⚡ Amplified quiet audio chunk for detection")

            # Resample to 16 kHz for YAMNet
            audio_chunk = librosa.resample(audio_chunk, orig_sr=self.audio_samplerate, target_sr=16000)

            # Normalize to [-1,1] after resample
            audio_chunk = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-9)

            # Debug: check after resample
            rms_post = np.sqrt(np.mean(audio_chunk**2))
            peak_post = np.max(np.abs(audio_chunk))
            print(f"🔊 Resampled audio RMS: {rms_post:.4f}, peak: {peak_post:.4f}")

            return audio_chunk
      return None



    def start_camera_detection(self):
        """
        Main loop: capture frames, buffer sequences, run model on sequences,
        run audio detection on available audio chunks, update UI and alarms.
        """
        print("▶️ Starting camera detection. Press 'q' in the video window to stop.")
        # Start audio capture if available and configured
        if self.use_audio:
            self._start_audio_stream()
        else:
            if not SOUNDDEVICE_AVAILABLE:
                print("⚠️ sounddevice not installed — microphone detection disabled.")
            elif self.predictor.yamnet_model is None:
                print("⚠️ YAMNet not loaded — microphone detection disabled.")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps):
            fps = 20.0  # fallback if camera doesn't provide fps
        print(f"📷 Camera FPS (detected/fallback): {fps}")

        strip_height = 200
        max_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 20
        video_prob = 0

        # keep last predicted video label for decision
        last_video_label = None

        # Start main loop
        try:
            while True:
                if self._stop_flag.is_set():
                    break

                ret, frame_bgr = self.cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from camera.")
                    break

                # initialize previous frame
                if self.previous_frame is None:
                    self.previous_frame = frame_bgr.copy()

                # Preprocess for model
                normalized_frame = self.predictor.Pre_Process_Video(frame_bgr, self.previous_frame)
                self.previous_frame = frame_bgr.copy()

                # Append to frames buffer
                self.frames_buffer.append(normalized_frame)

                # When enough frames, run model
                if len(self.frames_buffer) == self.predictor.sequence_length:
                    frames_array = np.expand_dims(np.array(self.frames_buffer), axis=0)
                    try:
                        probabilities = self.predictor.model.predict(frames_array)[0]
                        predicted_label = np.argmax(probabilities)
                        probability = math.floor(np.max(probabilities) * 100)
                        # update last_video_label for decision usage
                        last_video_label = int(predicted_label)
                        # update messages same as file mode
                        self.predictor.generate_message_content(probability, predicted_label)
                        self.predictor.video_message = f"{self.predictor.video_message}: {probability}%"
                        video_prob = probability
                        # Clear buffer to create non-overlapping sequences similar to file code
                        self.frames_buffer.clear()
                    except Exception as e:
                        print("⚠️ Model prediction error (video):", e)

                # Audio: check if chunk available
                if self.use_audio:
                    audio_chunk = self._get_audio_chunk_if_available()
                    if audio_chunk is not None:
                        try:
                            _, _ = self.predictor.predict_audio_chunk(audio_chunk)
                        except Exception as e:
                            print("⚠️ Audio prediction error:", e)

                audio_suspicious = "Suspicious" in self.predictor.audio_message
                is_red = False
                # NEW: require label == 0 for video-prob-based red alerts
                if (video_prob > 80 and last_video_label == 0) or audio_suspicious:
                    alarm_message = "Signal is RED 🚨"
                    alarm_color = (0, 0, 255)
                    is_red = True
                elif video_prob > 60:
                    alarm_message = "Signal is YELLOW ⚠️"
                    alarm_color = (0, 255, 255)
                else:
                    alarm_message = "Signal is GREEN ✅"
                    alarm_color = (0, 200, 0)

                if is_red:
                    if not self.alarm_triggered:
                        self.alarm_triggered = True
                        self.predictor.alarm_triggered = True
                        self.predictor.start_beeping()
                else:
                    if self.alarm_triggered:
                        self.alarm_triggered = False
                        self.predictor.alarm_triggered = False

                # Build bottom information strip
                strip = np.ones((strip_height, frame_bgr.shape[1], 3), dtype=np.uint8) * 255
                self.predictor.draw_text(strip, f"Video Log: {self.predictor.video_message}", (10, 30),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, max_width)
                self.predictor.draw_text(strip, f"Audio Log: {self.predictor.audio_message}", (10, 90),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, max_width)
                self.predictor.draw_text(strip, f"System Status: {alarm_message}", (10, 150),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, alarm_color, 3, max_width)

                combined = np.vstack((frame_bgr, strip))
                # Show combined
                cv2.imshow("Live Shoplifting Monitor", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Stop requested by user (q pressed).")
                    break

        except KeyboardInterrupt:
            print("🛑 Keyboard interrupt received. Stopping.")
        finally:
            # cleanup
            self._stop_audio_stream()
            self.cap.release()
            cv2.destroyAllWindows()
            self.predictor.alarm_triggered = False
            self.alarm_triggered = False
            print("🔚 Camera detection stopped.")
