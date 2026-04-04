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

# ── Tabs: Image Captioning | Video Captioning ────────────────────
tab_image, tab_video = st.tabs(["🖼️ Image Captioning", "🎬 Video Captioning"])

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
                "Improves caption coherence across frames:\n\n"
                "**Option A** — each frame is captioned with awareness of "
                "the previous 2 captions (prompt-based context).\n\n"
                "**Option B** — final summary uses semantic deduplication "
                "(SBERT) + T5 summarisation instead of simple joining.\n\n"
                "⚠️ First run downloads ~100 MB of models."
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
