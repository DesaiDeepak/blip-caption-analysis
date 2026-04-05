"""
Video Captioning Module
───────────────────────
Extracts frames from a video, generates BLIP captions per frame,
removes duplicates, and aggregates them into a coherent temporal
paragraph.

Usage:
    from video_captioning import VideoCaptioner
    vc = VideoCaptioner(model_dir="saved_models/fine_tuned_blip")
    result = vc.process_video("path/to/video.mp4", fps=1, max_frames=30)
    # result keys: frames, captions, unique_captions, aggregated_caption
"""

import os
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from utils.logger import get_logger

logger = get_logger("video_captioning")

# ── Temporal connectors used to stitch captions ──────────────────────
_TEMPORAL_CONNECTORS = [
    "Initially",
    "Then",
    "After that",
    "Subsequently",
    "Next",
    "Later",
    "Following that",
    "Afterwards",
    "Eventually",
    "Finally",
]


class VideoCaptioner:
    """End-to-end video → captions → aggregated paragraph pipeline."""

    def __init__(
        self,
        model_dir: str = "saved_models/fine_tuned_blip",
        fallback_model: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
    ):
        # ── Resolve device ────────────────────────────────────────
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        # ── Load model once ───────────────────────────────────────
        if os.path.isdir(model_dir) and os.path.isfile(
            os.path.join(model_dir, "config.json")
        ):
            source = model_dir
            logger.info(f"Loading fine-tuned BLIP from {source}")
        else:
            source = fallback_model
            logger.warning(
                f"Fine-tuned model not found at '{model_dir}'. "
                f"Falling back to {fallback_model}"
            )

        self.processor = BlipProcessor.from_pretrained(source)
        self.model = BlipForConditionalGeneration.from_pretrained(source)
        self.model.to(self._device)
        self.model.eval()
        logger.info(f"BLIP model loaded on {self._device}")

    # ── Frame extraction ──────────────────────────────────────────

    @staticmethod
    def extract_frames(
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 30,
    ) -> Tuple[List[np.ndarray], float, int]:
        """
        Extract frames from *video_path* at the requested sampling rate.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        fps : float
            Desired frames per second to sample (default 1).
        max_frames : int
            Hard cap on total frames returned (default 30).

        Returns
        -------
        frames : list[np.ndarray]
            BGR frames (OpenCV format).
        video_fps : float
            Original FPS of the video.
        total_video_frames : int
            Total frame count of the source video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Interval in *source* frame indices between two sampled frames
        interval = max(1, int(round(video_fps / fps)))

        frames: List[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                frames.append(frame)
                if len(frames) >= max_frames:
                    break
            frame_idx += 1

        cap.release()
        logger.info(
            f"Extracted {len(frames)} frames "
            f"(video FPS={video_fps:.1f}, interval={interval}, "
            f"total source frames={total_video_frames})"
        )
        return frames, video_fps, total_video_frames

    # ── Single-frame captioning ───────────────────────────────────

    def caption_frame(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str = "",
    ) -> str:
        """Generate a BLIP caption for a single BGR (OpenCV) frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR frame from OpenCV.
        text_prompt : str, optional
            Base prompt for conditional generation (e.g. medical mode).

        Notes
        -----
        Temporal context (awareness of previous captions) is handled
        exclusively in the *post-processing* step via SBERT semantic
        dedup + connector-join aggregation.  Injecting prior captions
        into the BLIP prompt causes exponential echo/growth and is
        therefore **not** done here.
        """
        # OpenCV → PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        effective_prompt = text_prompt.strip()

        if effective_prompt:
            inputs = self.processor(
                images=pil_image, text=effective_prompt, return_tensors="pt"
            )
        else:
            inputs = self.processor(images=pil_image, return_tensors="pt")

        inputs = {
            k: v.to(self._device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(output[0], skip_special_tokens=True).strip()

        # For conditional generation BLIP echoes the prompt prefix — strip it.
        if effective_prompt:
            prompt_lower = effective_prompt.lower().strip()
            caption_lower = caption.lower()
            if caption_lower.startswith(prompt_lower):
                caption = caption[len(effective_prompt):].strip().lstrip(":").strip()

        return caption

    # ── Duplicate removal ─────────────────────────────────────────

    @staticmethod
    def remove_duplicate_captions(
        captions: List[str],
    ) -> Tuple[List[str], List[int]]:
        """
        Remove consecutive duplicate captions while preserving order.

        Returns
        -------
        unique : list[str]
            De-duplicated captions.
        indices : list[int]
            Original indices of the kept captions.
        """
        if not captions:
            return [], []

        unique: List[str] = [captions[0]]
        indices: List[int] = [0]

        for i, cap in enumerate(captions[1:], start=1):
            if cap.lower().strip() != unique[-1].lower().strip():
                unique.append(cap)
                indices.append(i)

        return unique, indices

    # ── Temporal aggregation ──────────────────────────────────────

    @staticmethod
    def aggregate_captions(unique_captions: List[str], max_events: int = 8) -> str:
        """
        Stitch unique captions into a coherent temporal paragraph using
        connectors like "Initially", "Then", "After that", "Finally".

        If more than ``max_events`` captions remain after dedup, evenly
        samples ``max_events`` key moments so the paragraph stays readable.
        """
        if not unique_captions:
            return ""
        if len(unique_captions) == 1:
            return unique_captions[0].capitalize()

        # If too many captions remain, sample evenly spaced key moments
        if len(unique_captions) > max_events:
            indices = [
                int(round(i * (len(unique_captions) - 1) / (max_events - 1)))
                for i in range(max_events)
            ]
            unique_captions = [unique_captions[i] for i in indices]

        parts: List[str] = []
        n = len(unique_captions)

        for i, cap in enumerate(unique_captions):
            # Strip trailing period for cleaner joining
            cap_clean = cap.strip().rstrip(".")

            if i == 0:
                connector = _TEMPORAL_CONNECTORS[0]  # "Initially"
            elif i == n - 1:
                connector = _TEMPORAL_CONNECTORS[-1]  # "Finally"
            else:
                # Cycle through the middle connectors
                mid_idx = ((i - 1) % (len(_TEMPORAL_CONNECTORS) - 2)) + 1
                connector = _TEMPORAL_CONNECTORS[mid_idx]

            # Lowercase the first letter of caption after the connector
            if cap_clean:
                cap_clean = cap_clean[0].lower() + cap_clean[1:]

            parts.append(f"{connector}, {cap_clean}")

        return ". ".join(parts) + "."

    # ── Full pipeline ─────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        fps: float = 1.0,
        max_frames: int = 30,
        progress_callback=None,
        text_prompt: str = "",
        use_temporal_context: bool = False,
    ) -> dict:
        """
        Run the complete video-captioning pipeline.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        fps : float
            Sampling rate in frames per second.
        max_frames : int
            Maximum frames to process.
        progress_callback : callable, optional
            ``callback(current_frame: int, total_frames: int)``
            invoked after each frame is captioned.
        text_prompt : str, optional
            Base prompt for conditional generation (medical mode etc.).
        use_temporal_context : bool
            When True, enables tighter semantic dedup (threshold=0.75)
            producing fewer, cleaner key-event sentences.  When False,
            a looser threshold (0.85) keeps more scene variety.
            Both modes use SBERT dedup + connector-join aggregation.

        Returns
        -------
        dict with keys:
            frames                  – list of BGR numpy arrays
            frame_images            – list of PIL RGB images (for display)
            captions                – list[str], one per frame
            unique_captions         – list[str], consecutive duplicates removed
            semantic_unique_captions – list[str], semantically deduped
              (equals unique_captions when use_temporal_context=False)
            unique_indices          – list[int], frame indices of unique captions
            aggregated_caption      – str, final temporal paragraph
            video_fps               – float, original video FPS
            total_video_frames      – int, source video frame count
            temporal_context_used   – bool, mirrors use_temporal_context flag
        """
        # 1. Extract frames
        frames, video_fps, total_video_frames = self.extract_frames(
            video_path, fps=fps, max_frames=max_frames
        )

        if not frames:
            raise ValueError("No frames could be extracted from the video.")

        # Lazy-load TemporalFusion only when needed (avoids model download on
        # every import).  Note: temporal_fusion is also loaded for non-temporal
        # mode below (step 4) for SBERT dedup.
        temporal_fusion = None

        # 2. Caption each frame (unconditional — no prompt injection).
        # Temporal awareness is handled in step 4 via SBERT dedup, not
        # by feeding prior captions into the BLIP prompt.
        captions: List[str] = []
        frame_images: List[Image.Image] = []
        skipped_black = 0
        skipped_lowinfo = 0

        for i, frame in enumerate(frames):
            # Skip pure-black frames (title cards, fade-in/out, intro screens).
            # A frame whose mean grayscale brightness is below 15 (out of 255)
            # is essentially black and will only produce "a black background" captions.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.mean() < 15:
                skipped_black += 1
                logger.debug(f"Frame {i + 1}: skipped (black frame, mean={gray.mean():.1f})")
                if progress_callback is not None:
                    progress_callback(i + 1, len(frames))
                continue

            # Skip logo / title-card / solid-color frames.
            # These have very low edge energy (Laplacian variance < 100)
            # compared to real-world scenes which typically score 200+.
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var < 100:
                skipped_lowinfo += 1
                logger.debug(f"Frame {i + 1}: skipped (low detail, laplacian_var={lap_var:.0f})")
                if progress_callback is not None:
                    progress_callback(i + 1, len(frames))
                continue

            caption = self.caption_frame(
                frame,
                text_prompt=text_prompt,
            )
            captions.append(caption)

            # Convert for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_images.append(Image.fromarray(rgb))

            if progress_callback is not None:
                progress_callback(i + 1, len(frames))

            logger.info(f"Frame {i + 1}/{len(frames)}: {caption}")

        if skipped_black or skipped_lowinfo:
            logger.info(
                f"Skipped frames — black: {skipped_black}, "
                f"low-detail (logos/titles): {skipped_lowinfo}"
            )

        # 3. Remove consecutive exact duplicates (always done)
        unique_captions, unique_indices = self.remove_duplicate_captions(captions)

        # 4. Aggregate
        # Lazy-load TemporalFusion with mode-specific dedup threshold:
        #   • Temporal ON  → threshold=0.75  (tighter: fewer, cleaner key events)
        #   • Temporal OFF → threshold=0.85  (looser: preserves more scene variety)
        if temporal_fusion is None:
            from temporal_fusion import TemporalFusion
            dedup_threshold = 0.75 if use_temporal_context else 0.85
            temporal_fusion = TemporalFusion(dedup_threshold=dedup_threshold)

        semantic_unique = temporal_fusion.semantic_dedup(unique_captions)

        # Both modes use connector-join for aggregation.
        # T5 summarisation was removed because it over-condensed 11 captions
        # into only 2-3 sentences, losing important scene detail.
        aggregated = self.aggregate_captions(semantic_unique)

        logger.info(
            f"Pipeline complete — {len(frames)} frames, "
            f"{len(unique_captions)} exact-unique, "
            f"{len(semantic_unique)} semantic-unique captions "
            f"(temporal_context={use_temporal_context})"
        )

        return {
            "frames": frames,
            "frame_images": frame_images,
            "captions": captions,
            "unique_captions": unique_captions,
            "semantic_unique_captions": semantic_unique,
            "unique_indices": unique_indices,
            "aggregated_caption": aggregated,
            "video_fps": video_fps,
            "total_video_frames": total_video_frames,
            "temporal_context_used": use_temporal_context,
        }
