import os
import tempfile
import requests
import streamlit as st
from dotenv import load_dotenv
from medical_alert import detect_medical_issue

load_dotenv()


API_KEY = os.environ.get("API_KEY")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_DIR = "saved_models/fine_tuned_blip"
IMAGES_FOLDER = "./Images/Images"
CAPTIONS_FILE = "./Images/captions.txt"
CLEANED_CAPTIONS_FILE = "./Images/cleaned_captions.txt"


def check_api_healthy():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200 and response.json().get("model_loaded")
    except requests.exceptions.ConnectionError:
        return False


# ── Page config ───────────────────────────────────────────────────
st.set_page_config(page_title="BLIP Captioning App", layout="wide")

# If no trained model exists, show an info banner (the API server
# already falls back to the public base model automatically).
if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
    st.info(
        "ℹ️ No fine-tuned model found at `saved_models/fine_tuned_blip/`. "
        "The API is using the public **Salesforce/blip-image-captioning-base** model. "
        "To train your own, run:  \n"
        "`python -c \"from training.model_training import train_model; "
        "train_model('./Images/cleaned_captions.txt', './Images/Images', epochs=5)\"`"
    )


# Title of the application
st.title("BLIP Captioning App")
st.write("Upload an image or video, and I'll generate captions!")

# ── Tabs: Image Captioning | Video Captioning | Live Captioning ──
tab_image, tab_video, tab_live = st.tabs(
    ["🖼️ Image Captioning", "🎬 Video Captioning", "📹 Live Captioning"]
)

# ══════════════════════════════════════════════════════════════════
#  TAB 1 — IMAGE CAPTIONING  (original pipeline, untouched logic)
# ══════════════════════════════════════════════════════════════════
with tab_image:
    st.header("Image Captioning")
    st.write("Upload an image, and I'll generate a caption for it!")

    # Check API connectivity
    if not check_api_healthy():
        st.error(
            "API server is not reachable. Please start it with: "
            "`uvicorn api.inference:app --port 8000`"
        )
    else:
        # ── Medical Image Mode toggle ─────────────────────────
        medical_mode = st.checkbox(
            "🩺 Medical Image Mode",
            value=False,
            help="Enable this for X-rays, CT scans, MRIs, etc. "
            "Uses prompt-guided captioning to generate medically "
            "relevant descriptions.",
        )

        # File uploader widget for image input
        uploaded_image = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Build request payload
            prompt = "a medical image showing" if medical_mode else ""

            # Send image to FastAPI and get caption
            with st.spinner("Generating caption..."):
                response = requests.post(
                    f"{API_URL}/caption",
                    files={
                        "file": (
                            uploaded_image.name,
                            uploaded_image.getvalue(),
                            uploaded_image.type,
                        )
                    },
                    data={"text_prompt": prompt},
                    headers={"X-API-Key": API_KEY},
                )

            if response.status_code == 200:
                caption = response.json()["caption"]
                st.subheader("Generated Caption:")
                st.write(caption)

                # ── Medical Alerting Layer ────────────────────
                st.markdown("---")
                st.subheader("🩺 Medical Analysis")
                status, explanation, confidence = detect_medical_issue(caption)

                if status == "CRITICAL":
                    st.error(f"**Status: {status}**")
                elif status == "WARNING":
                    st.warning(f"**Status: {status}**")
                elif status == "REVIEW":
                    st.info(f"**Status: {status}**")
                else:
                    st.success(f"**Status: {status}**")

                st.write(explanation)
                st.caption("Confidence")
                st.progress(confidence)
            else:
                st.error(
                    f"Error from API: "
                    f"{response.json().get('detail', 'Unknown error')}"
                )
        else:
            st.write("Please upload an image to generate a caption.")

# ══════════════════════════════════════════════════════════════════
#  TAB 2 — VIDEO CAPTIONING  (new feature)
# ══════════════════════════════════════════════════════════════════
with tab_video:
    st.header("Video Captioning")
    st.write(
        "Upload a video and I'll extract frames, generate per-frame "
        "captions, and produce a coherent temporal summary."
    )

    # ── Sidebar-style controls inside the tab ─────────────────────
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ Settings")
        sample_fps = st.slider(
            "Frame sampling rate (FPS)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="How many frames per second to sample from the video.",
        )
        max_frames = st.number_input(
            "Max frames to process",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            help="Hard cap on the number of frames to avoid overload.",
        )
        video_medical_mode = st.checkbox(
            "🩺 Medical Video Mode",
            value=False,
            help="Enable for medical videos (endoscopy, ultrasound, "
            "surgical footage, etc.). Uses prompt-guided captioning.",
        )
        use_temporal_context = st.checkbox(
            "🕐 Temporal Context Mode",
            value=False,
            help=(
                "Enables tighter semantic deduplication (SBERT) so the "
                "final summary keeps only the most distinct scene changes, "
                "producing a cleaner, more focused temporal narrative.\n\n"
                "⚠️ First run downloads ~80 MB of SBERT model."
            ),
        )

    with col_main:
        uploaded_video = st.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov"]
        )

    if uploaded_video is not None:
        # ── Video preview ─────────────────────────────────────
        st.video(uploaded_video)

        # ── Save to temp file for OpenCV ──────────────────────
        suffix = os.path.splitext(uploaded_video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.getvalue())
            tmp_path = tmp.name

        # ── Process button ────────────────────────────────────
        if st.button("🚀 Generate Video Captions", type="primary"):
            try:
                # Lazy-load the captioner so the model is only loaded
                # when the user actually clicks the button.  Cache it
                # in session_state so subsequent runs are instant.
                if "video_captioner" not in st.session_state:
                    with st.spinner("Loading BLIP model for video captioning…"):
                        from video_captioning import VideoCaptioner

                        st.session_state.video_captioner = VideoCaptioner(
                            model_dir=MODEL_DIR
                        )

                captioner = st.session_state.video_captioner

                # ── Progress bar ──────────────────────────────
                progress_bar = st.progress(0, text="Extracting & captioning frames…")

                def _update_progress(current: int, total: int):
                    pct = current / total
                    progress_bar.progress(
                        pct, text=f"Captioning frame {current}/{total}…"
                    )

                result = captioner.process_video(
                    video_path=tmp_path,
                    fps=sample_fps,
                    max_frames=int(max_frames),
                    progress_callback=_update_progress,
                    text_prompt="a medical image showing" if video_medical_mode else "",
                    use_temporal_context=use_temporal_context,
                )

                progress_bar.progress(1.0, text="✅ Done!")

                # ── Stats ─────────────────────────────────────
                semantic_unique = result.get(
                    "semantic_unique_captions", result["unique_captions"]
                )
                stats_msg = (
                    f"📊 Processed **{len(result['captions'])}** frames "
                    f"({len(result['unique_captions'])} exact-unique"
                )
                if use_temporal_context:
                    stats_msg += (
                        f", **{len(semantic_unique)} semantic-unique** after SBERT dedup"
                    )
                stats_msg += (
                    f") from a {result['total_video_frames']}-frame video "
                    f"@ {result['video_fps']:.1f} FPS"
                )
                st.info(stats_msg)

                # ── Aggregated caption ────────────────────────
                st.markdown("---")
                st.subheader("📝 Summarized Caption")
                st.markdown(
                    f"> {result['aggregated_caption']}"
                )

                # ── Medical interpretation ────────────────────
                st.markdown("---")
                st.subheader("🩺 Medical Analysis")
                status, explanation, confidence = detect_medical_issue(
                    result["aggregated_caption"]
                )

                if status == "CRITICAL":
                    st.error(f"**Status: {status}**")
                elif status == "WARNING":
                    st.warning(f"**Status: {status}**")
                elif status == "REVIEW":
                    st.info(f"**Status: {status}**")
                else:
                    st.success(f"**Status: {status}**")

                st.write(explanation)
                st.caption("Confidence")
                st.progress(confidence)

                # ── Frame-by-frame captions (expandable) ──────
                st.markdown("---")
                with st.expander(
                    f"🎞️ Frame-by-Frame Captions ({len(result['captions'])} frames)",
                    expanded=False,
                ):
                    # Show frames in a grid (3 columns)
                    cols = st.columns(3)
                    for idx, (img, cap) in enumerate(
                        zip(result["frame_images"], result["captions"])
                    ):
                        with cols[idx % 3]:
                            st.image(
                                img,
                                caption=f"Frame {idx + 1}",
                                use_column_width=True,
                            )
                            st.caption(cap)

            except Exception as exc:
                st.error(f"Error processing video: {exc}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    else:
        st.write("Please upload a video file to get started.")

# ══════════════════════════════════════════════════════════════════
#  TAB 3 — LIVE CAPTIONING  (webcam / screen capture)
# ══════════════════════════════════════════════════════════════════
with tab_live:
    st.header("Live Captioning")
    st.write(
        "Capture frames from your webcam and see BLIP captions "
        "update in near-real-time.  Captions are semantically "
        "deduplicated so repeated scenes don't clutter the output."
    )

    # ── Controls ──────────────────────────────────────────────────
    live_col_ctrl, live_col_main = st.columns([1, 3])

    with live_col_ctrl:
        st.subheader("⚙️ Settings")
        caption_interval = st.slider(
            "Caption every N seconds",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="How often to generate a new caption from the webcam feed.",
        )
        max_history = st.slider(
            "Caption history size",
            min_value=5,
            max_value=30,
            value=15,
            step=5,
            help="Maximum number of captions to keep in the live feed.",
        )
        live_medical_mode = st.checkbox(
            "🩺 Medical Mode",
            value=False,
            help="Use prompt-guided captioning for medical imagery.",
            key="live_medical",
        )

    # ── Initialise session state for live captioning ──────────────
    if "live_running" not in st.session_state:
        st.session_state.live_running = False
    if "live_captions" not in st.session_state:
        st.session_state.live_captions = []

    with live_col_main:
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_btn = st.button("▶️ Start Live Captioning", type="primary")
        with btn_col2:
            stop_btn = st.button("⏹️ Stop")
        with btn_col3:
            clear_btn = st.button("🗑️ Clear History")

    if clear_btn:
        st.session_state.live_captions = []
        st.rerun()

    if stop_btn:
        st.session_state.live_running = False
        st.rerun()

    if start_btn:
        st.session_state.live_running = True

        # Lazy-load the captioner (reuse from video tab if available)
        if "video_captioner" not in st.session_state:
            with st.spinner("Loading BLIP model…"):
                from video_captioning import VideoCaptioner
                st.session_state.video_captioner = VideoCaptioner(
                    model_dir=MODEL_DIR
                )
        captioner = st.session_state.video_captioner

        # Lazy-load SBERT for live dedup
        if "live_tf" not in st.session_state:
            with st.spinner("Loading SBERT for dedup…"):
                from temporal_fusion import TemporalFusion
                st.session_state.live_tf = TemporalFusion(dedup_threshold=0.80)
                st.session_state.live_tf._load_sbert()  # eagerly load

        tf = st.session_state.live_tf

        import cv2
        import time
        import numpy as np

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error(
                "❌ Could not open webcam. Please check that:\n"
                "- Your browser has granted camera permissions\n"
                "- No other application is using the camera\n"
                "- You are running Streamlit locally (not via remote SSH)"
            )
            st.session_state.live_running = False
        else:
            st.success("📹 Webcam connected — generating captions…")

            # Layout: live frame on left, caption feed on right
            frame_placeholder = st.empty()
            caption_placeholder = st.empty()
            status_placeholder = st.empty()

            text_prompt = "a medical image showing" if live_medical_mode else ""
            last_caption_time = 0

            try:
                while st.session_state.live_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Lost webcam feed.")
                        break

                    # Show the live frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, caption="Live Feed", width=480
                    )

                    # Caption at the configured interval
                    now = time.time()
                    if now - last_caption_time >= caption_interval:
                        # Skip dark frames
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if gray.mean() < 15:
                            status_placeholder.caption("⏭️ Skipped dark frame")
                            last_caption_time = now
                            continue

                        with status_placeholder:
                            st.caption("🔄 Generating caption…")

                        caption = captioner.caption_frame(
                            frame, text_prompt=text_prompt
                        )

                        # SBERT dedup against recent captions
                        recent = st.session_state.live_captions[-5:]
                        is_duplicate = False
                        if recent and tf._sbert is not None:
                            try:
                                from sentence_transformers import util
                                new_emb = tf._sbert.encode(
                                    caption, convert_to_tensor=True
                                )
                                for prev in recent:
                                    prev_emb = tf._sbert.encode(
                                        prev, convert_to_tensor=True
                                    )
                                    sim = util.cos_sim(new_emb, prev_emb).item()
                                    if sim >= 0.80:
                                        is_duplicate = True
                                        break
                            except Exception:
                                pass

                        if not is_duplicate:
                            st.session_state.live_captions.append(caption)
                            # Trim to max_history
                            if len(st.session_state.live_captions) > max_history:
                                st.session_state.live_captions = (
                                    st.session_state.live_captions[-max_history:]
                                )

                        last_caption_time = now

                        # Update the caption feed
                        with caption_placeholder.container():
                            st.subheader(
                                f"📝 Live Captions "
                                f"({len(st.session_state.live_captions)})"
                            )
                            # Show latest caption prominently
                            if st.session_state.live_captions:
                                st.markdown(
                                    f"**Latest:** {st.session_state.live_captions[-1]}"
                                )
                            # Show history
                            for i, c in enumerate(
                                reversed(st.session_state.live_captions[:-1]), 1
                            ):
                                st.caption(f"{i}. {c}")

                        status_placeholder.caption(
                            f"✅ Caption generated"
                            + (" (duplicate skipped)" if is_duplicate else "")
                        )

                    # Small sleep to avoid busy-looping
                    time.sleep(0.1)

            except Exception as exc:
                st.error(f"Live captioning error: {exc}")
            finally:
                cap.release()
                st.session_state.live_running = False

    # Show existing caption history even when not running
    elif st.session_state.live_captions:
        st.subheader(
            f"📝 Caption History ({len(st.session_state.live_captions)})"
        )
        for i, c in enumerate(st.session_state.live_captions, 1):
            st.caption(f"{i}. {c}")
