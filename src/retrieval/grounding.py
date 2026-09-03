"""GroundingDINO wrapper: detect a named subject and crop the image.

Integrated into ``search_by_image`` when ``visual_mode`` is ``crop_only``
or ``both``.  In those modes the agent passes the entity name it identified;
GroundingDINO locates that subject in the image and crops to it so that
EVA-CLIP focuses on the subject rather than the full scene.

The model is loaded lazily on first call and shared across tool invocations
within a single run.

The crop is always rescaled back to the original image size before being
passed to EVA-CLIP: this ensures the encoder allocates the same number of
visual tokens and that the crop similarity scores are directly comparable
to full-image scores in the RRF merge step.
"""

from __future__ import annotations

import threading

import torch
from PIL import Image

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25
_PAD_RATIO = 0.08   # padding fraction around the detected box


class Grounder:
    """Detect a named entity in an image with GroundingDINO and crop to it.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.  Defaults to ``grounding-dino-tiny``
        which is fast and accurate enough for landmark / species detection.
    device:
        Torch device.  Defaults to CPU to avoid VRAM contention with the
        VLM and EVA-CLIP that may already occupy the GPU.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu"):
        self.model_id = model_id
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            # Import here so the module can be imported without transformers
            # installed — only callers that actually use detection pay the cost.
            from transformers import (
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
            )
            print(f"Loading GroundingDINO ({self.model_id}) on {self.device} …")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )
            print("GroundingDINO ready.")

    def detect(
        self,
        image: Image.Image,
        query: str,
        box_threshold: float = _BOX_THRESHOLD,
        text_threshold: float = _TEXT_THRESHOLD,
    ) -> list[tuple[float, list[float]]]:
        """Detect ``query`` in ``image``; return ``(score, box)`` pairs, best first.

        ``box`` is ``[x0, y0, x1, y1]`` in absolute pixel coordinates.
        Returns an empty list when nothing is found above the thresholds.
        """
        self._ensure_model()
        # GroundingDINO requires the prompt to end with a period.
        text = query.rstrip(".") + "."
        inputs = self._processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        w, h = image.size
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )[0]
        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        return sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)

    def crop(
        self,
        image: Image.Image,
        query: str,
        pad: float = _PAD_RATIO,
    ) -> Image.Image | None:
        """Crop and upscale to the best detection of ``query`` in ``image``.

        Adds a small margin (``pad`` fraction of box dimensions) so the
        subject is not clipped at the edges.  The crop is then rescaled
        to the original image size with bicubic interpolation so that
        EVA-CLIP sees the same pixel dimensions and the resulting
        embedding is comparable to the full-image embedding.

        Returns ``None`` when GroundingDINO finds nothing above threshold.
        """
        hits = self.detect(image, query)
        if not hits:
            return None
        _, box = hits[0]
        x0, y0, x1, y1 = box
        w, h = image.size
        pw = (x1 - x0) * pad
        ph = (y1 - y0) * pad
        x0 = max(0.0, x0 - pw)
        y0 = max(0.0, y0 - ph)
        x1 = min(float(w), x1 + pw)
        y1 = min(float(h), y1 + ph)
        return image.crop((x0, y0, x1, y1)).resize(image.size, Image.BICUBIC)
