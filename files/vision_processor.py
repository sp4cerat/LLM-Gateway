"""
LLM Gateway v1.5 — Vision Preprocessor

Tiered Vision Pipeline for cost-optimized image processing:

  Tier 0: Image Preprocessing (resize, optimize, grayscale for OCR)
  Tier 1: PaddleOCR (free, CPU, ~1-2s) — text extraction from documents
          Fallback: Tesseract OCR if PaddleOCR unavailable
  Tier 2: Ollama/Moondream (local, CPU, ~2-4s) — object/scene description
  Tier 3: Cloud Vision LLM (paid) — complex reasoning, fallback

Flow:
  Image → Intent Detection → Preprocess → OCR/Local Vision → Route Decision
    → Either: text-only to cheap model (90% cost savings)
    → Or: image passthrough to medium/premium vision model
"""

import asyncio
import base64
import io
import logging
import os
from enum import Enum
from typing import Optional

# Skip PaddlePaddle's slow model hoster connectivity check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Disable MKL-DNN — crashes on some VMs with ConvertPirAttribute2RuntimeAttribute error
os.environ["FLAGS_use_mkldnn"] = "0"

log = logging.getLogger("gateway.vision")


# ═══════════════════════════════════════════════════════════════════════════════
#  Multimodal Intent Detection (text-only, no image needed)
# ═══════════════════════════════════════════════════════════════════════════════

class VisionIntent(str, Enum):
    TEXT_EXTRACT = "text_extract"      # OCR: "lies vor", "was steht da", "Rechnung"
    OBJECT_SIMPLE = "object_simple"    # Simple: "was ist das", "welches Tier"
    COMPLEX_VISION = "complex_vision"  # Complex: "erkläre die Stimmung", "finde Fehler"
    CODE_SCREENSHOT = "code_screenshot"  # Code in image: "Fehlermeldung", "Code"
    UNKNOWN = "unknown"


# Keywords for intent detection (DE + EN)
_OCR_KEYWORDS = {
    "lies", "lese", "lesen", "text", "steht da", "steht dort", "abschreiben",
    "zusammenfassung", "rechnung", "dokument", "vertrag", "brief", "formular",
    "iban", "kontonummer", "adresse", "quittung", "beleg", "ausweis",
    "read", "extract", "transcribe", "document", "receipt", "invoice", "ocr",
    "scan", "what does it say", "was steht", "übersetze",
}

_SIMPLE_VISION_KEYWORDS = {
    "was ist das", "was siehst du", "was zeigt", "wer ist", "welches tier",
    "farbe", "objekt", "gegenstand", "erkennen", "identifiziere",
    "what is this", "what do you see", "identify", "what color", "describe",
    "recognize", "animal", "object", "person", "food", "plant",
}

_CODE_KEYWORDS = {
    "fehler", "fehlermeldung", "error", "traceback", "stacktrace", "exception",
    "code", "screenshot", "terminal", "konsole", "console", "debug", "bug",
    "syntax", "log", "ausgabe", "output",
}

_COMPLEX_KEYWORDS = {
    "erkläre", "analysiere", "vergleiche", "stimmung", "ironie", "meme",
    "diagramm", "chart", "graph", "tabelle", "layout", "design",
    "explain", "analyze", "compare", "mood", "irony", "diagram",
    "infographic", "interpret", "evaluate", "relationship",
}


def detect_vision_intent(query: str) -> VisionIntent:
    """Classify user intent for multimodal requests (text-only analysis)."""
    q = query.lower()

    # Check each category (order matters: specific → general)
    for kw in _CODE_KEYWORDS:
        if kw in q:
            return VisionIntent.CODE_SCREENSHOT

    for kw in _OCR_KEYWORDS:
        if kw in q:
            return VisionIntent.TEXT_EXTRACT

    for kw in _COMPLEX_KEYWORDS:
        if kw in q:
            return VisionIntent.COMPLEX_VISION

    for kw in _SIMPLE_VISION_KEYWORDS:
        if kw in q:
            return VisionIntent.OBJECT_SIMPLE

    # Default: if there's an image, assume they want description
    return VisionIntent.OBJECT_SIMPLE


# ═══════════════════════════════════════════════════════════════════════════════
#  Image Preprocessing (resize, optimize, convert)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_pillow():
    """Import Pillow, return (Image, ImageOps) or (None, None)."""
    try:
        from PIL import Image, ImageOps
        return Image, ImageOps
    except ImportError:
        log.warning("Pillow not installed. Run: pip install Pillow")
        return None, None


def preprocess_image(
    image_b64: str,
    max_pixels: int = 1560,
    grayscale: bool = False,
    quality: int = 85,
    output_format: str = "JPEG",
) -> tuple[str, dict]:
    """
    Preprocess a base64 image for optimal LLM consumption.
    
    Returns:
        (processed_b64, info_dict) where info contains dimensions, size savings etc.
    """
    Image, ImageOps = _ensure_pillow()
    if Image is None:
        return image_b64, {"preprocessed": False, "reason": "pillow_not_installed"}

    try:
        raw_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw_bytes))
        original_size = len(raw_bytes)
        original_dims = img.size

        # Convert RGBA → RGB (for JPEG)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Grayscale for OCR (saves tokens, improves OCR accuracy)
        if grayscale:
            img = img.convert("L")
            img = ImageOps.autocontrast(img)

        # Resize if needed
        resized = False
        if max(img.size) > max_pixels:
            img.thumbnail((max_pixels, max_pixels), Image.Resampling.LANCZOS)
            resized = True

        # Save to buffer
        buf = io.BytesIO()
        save_format = output_format if not grayscale else "JPEG"
        img.save(buf, format=save_format, quality=quality, optimize=True)
        processed_bytes = buf.getvalue()
        processed_b64 = base64.b64encode(processed_bytes).decode()

        new_size = len(processed_bytes)
        savings = max(0, (1 - new_size / original_size)) * 100

        info = {
            "preprocessed": True,
            "original_dims": original_dims,
            "new_dims": img.size,
            "resized": resized,
            "grayscale": grayscale,
            "original_bytes": original_size,
            "new_bytes": new_size,
            "savings_pct": round(savings, 1),
            "format": save_format,
        }
        log.info(f"Image preprocessed: {original_dims}→{img.size} | "
                 f"{original_size//1024}KB→{new_size//1024}KB ({savings:.0f}% saved)")
        return processed_b64, info

    except Exception as e:
        log.warning(f"Image preprocessing failed: {e}")
        return image_b64, {"preprocessed": False, "reason": str(e)}


def preprocess_for_ocr(image_b64: str) -> str:
    """Preprocess image specifically for OCR (grayscale, high contrast, 1500px)."""
    processed, _ = preprocess_image(
        image_b64, max_pixels=1500, grayscale=True, quality=95
    )
    return processed


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 1: PaddleOCR (free, fast, CPU, better accuracy than Tesseract)
# ═══════════════════════════════════════════════════════════════════════════════

# Lazy-loaded PaddleOCR model (loads on first use, ~300MB download)
_paddle_model = None
_paddle_lock = None


def _get_paddle_ocr():
    """Lazy-load PaddleOCR model. Downloads models on first use."""
    global _paddle_model, _paddle_lock
    import threading
    if _paddle_lock is None:
        _paddle_lock = threading.Lock()

    if _paddle_model is not None:
        return _paddle_model

    with _paddle_lock:
        if _paddle_model is not None:
            return _paddle_model
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True: detect rotated text
            # lang='de': German + English support
            # use_gpu=False: CPU-only (safe for 8GB VPS)
            # show_log=False: suppress paddle's verbose logging
            log.info("Loading PaddleOCR model...")
            _paddle_model = PaddleOCR(
                use_angle_cls=True,
                lang='german',
                use_gpu=False,
                show_log=False,
            )
            log.info("PaddleOCR model loaded")
            return _paddle_model
        except ImportError:
            log.warning("PaddleOCR not installed. Run: pip install paddlepaddle paddleocr --break-system-packages")
            return None
        except Exception as e:
            log.error(f"Failed to load PaddleOCR: {e}")
            return None


async def paddle_ocr(image_b64: str) -> dict:
    """
    Extract text from image using PaddleOCR.

    Returns:
        {"success": bool, "text": str, "confidence": float, "method": "paddleocr"}
    """
    model = _get_paddle_ocr()
    if model is None:
        # Fallback to Tesseract if PaddleOCR unavailable
        return await _tesseract_ocr_fallback(image_b64)

    Image, _ = _ensure_pillow()
    if Image is None:
        return {"success": False, "text": "", "confidence": 0,
                "method": "paddleocr", "error": "Pillow not installed"}

    try:
        import numpy as np
        import traceback

        # PaddleOCR does its own preprocessing — don't convert to grayscale
        # Just resize if needed
        raw_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw_bytes))
        # Ensure RGB (PaddleOCR needs 3 channels)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Resize large images
        if max(img.size) > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)

        img_array = np.array(img)

        loop = asyncio.get_event_loop()

        def _run_ocr():
            # PaddleOCR v2.x uses .ocr(), v3.4+ uses .predict()
            try:
                if hasattr(model, 'ocr'):
                    raw_result = model.ocr(img_array, cls=True)
                elif hasattr(model, 'predict'):
                    raw_result = list(model.predict(img_array))
                else:
                    raise RuntimeError("No .ocr() or .predict() method")
            except Exception as e:
                log.error(f"PaddleOCR call failed: {e}\n{traceback.format_exc()}")
                raise

            texts = []
            confidences = []

            log.info(f"PaddleOCR raw: type={type(raw_result).__name__}, "
                     f"len={len(raw_result) if raw_result else 0}, "
                     f"preview={str(raw_result)[:500]}")

            if not raw_result:
                return "", 0

            # v2.x .ocr() returns: [[line1, line2, ...]] where each line = [box, (text, conf)]
            # First element is the page result
            page = raw_result[0] if raw_result else []
            if page is None:
                return "", 0

            for line in page:
                try:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        text_info = line[1]  # (text, confidence)
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            t, c = str(text_info[0]).strip(), float(text_info[1])
                            if t:
                                texts.append(t)
                                confidences.append(c)
                    elif isinstance(line, dict):
                        # v3.4 format
                        t = str(line.get('rec_text', line.get('text', ''))).strip()
                        c = float(line.get('rec_score', line.get('confidence', 0)))
                        if t:
                            texts.append(t)
                            confidences.append(c)
                except (ValueError, TypeError, IndexError) as e:
                    log.debug(f"PaddleOCR parse line error: {e}")
                    continue

            text = "\n".join(texts)
            avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0
            return text, avg_conf

        text, confidence = await loop.run_in_executor(None, _run_ocr)

        success = len(text) > 10 and confidence > 50
        log.info(f"PaddleOCR: {len(text)} chars, confidence={confidence:.0f}%, "
                 f"lines={len(text.splitlines())}, success={success}")

        return {
            "success": success,
            "text": text,
            "confidence": round(confidence, 1),
            "char_count": len(text),
            "method": "paddleocr",
        }

    except Exception as e:
        log.warning(f"PaddleOCR failed: {e}, trying Tesseract fallback")
        return await _tesseract_ocr_fallback(image_b64)


async def _tesseract_ocr_fallback(image_b64: str, languages: str = "deu+eng") -> dict:
    """Fallback to Tesseract if PaddleOCR is unavailable."""
    try:
        import pytesseract
    except ImportError:
        return {"success": False, "text": "", "confidence": 0,
                "method": "tesseract_fallback", "error": "neither paddleocr nor pytesseract installed"}

    Image, _ = _ensure_pillow()
    if Image is None:
        return {"success": False, "text": "", "confidence": 0,
                "method": "tesseract_fallback", "error": "Pillow not installed"}

    try:
        ocr_b64 = preprocess_for_ocr(image_b64)
        raw_bytes = base64.b64decode(ocr_b64)
        img = Image.open(io.BytesIO(raw_bytes))

        loop = asyncio.get_event_loop()

        def _run_ocr():
            data = pytesseract.image_to_data(img, lang=languages, output_type=pytesseract.Output.DICT)
            texts = []
            confidences = []
            for i, conf in enumerate(data["conf"]):
                if int(conf) > 0:
                    texts.append(data["text"][i])
                    confidences.append(int(conf))
            text = " ".join(t for t in texts if t.strip())
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            return text.strip(), avg_conf

        text, confidence = await loop.run_in_executor(None, _run_ocr)
        success = len(text) > 10 and confidence > 40
        log.info(f"Tesseract fallback: {len(text)} chars, confidence={confidence:.0f}%")

        return {
            "success": success,
            "text": text,
            "confidence": round(confidence, 1),
            "char_count": len(text),
            "method": "tesseract_fallback",
        }
    except Exception as e:
        log.warning(f"Tesseract fallback also failed: {e}")
        return {"success": False, "text": "", "confidence": 0,
                "method": "tesseract_fallback", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tier 2: Ollama Local Vision (Moondream2 / LLaVA)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaVision:
    """Local vision model via Ollama (Moondream2 recommended for 8GB VPS)."""

    def __init__(self, base_url: str = None, model: str = None, timeout: float = 30.0):
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_VISION_MODEL", "moondream")
        self.api_url = f"{self.base_url}/api/generate"
        self.timeout = timeout
        self._available = None  # Cached availability check

    async def is_available(self) -> bool:
        """Check if Ollama is running and model is loaded."""
        if self._available is not None:
            return self._available
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{self.base_url}/api/tags", timeout=3.0)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    self._available = any(self.model in m for m in models)
                    if self._available:
                        log.info(f"Ollama vision available: {self.model}")
                    else:
                        log.info(f"Ollama running but model '{self.model}' not found. "
                                 f"Available: {models}")
                else:
                    self._available = False
        except Exception:
            self._available = False
            log.info("Ollama not available (connection failed)")
        return self._available

    async def analyze(self, image_b64: str, query: str) -> dict:
        """
        Analyze image with local vision model.
        
        Returns:
            {"success": bool, "description": str, "model": str, "latency_ms": float}
        """
        if not await self.is_available():
            return {"success": False, "description": "",
                    "model": self.model, "error": "ollama_not_available"}

        # Preprocess: resize to 384px for speed (Moondream works fine at low res)
        processed_b64, prep_info = preprocess_image(
            image_b64, max_pixels=768, quality=80
        )

        prompt = f"Describe this image concisely, focusing on: {query}. " \
                 f"If there is any text visible, extract it exactly."

        try:
            import httpx
            import time
            start = time.time()

            async with httpx.AsyncClient() as client:
                r = await client.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [processed_b64],
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()

            latency = (time.time() - start) * 1000
            description = data.get("response", "").strip()

            log.info(f"Ollama vision ({self.model}): {len(description)} chars, "
                     f"{latency:.0f}ms")

            return {
                "success": bool(description),
                "description": description,
                "model": self.model,
                "latency_ms": round(latency, 1),
            }

        except Exception as e:
            log.warning(f"Ollama vision failed: {e}")
            return {"success": False, "description": "",
                    "model": self.model, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Vision Pipeline: Orchestrates all tiers
# ═══════════════════════════════════════════════════════════════════════════════

class VisionPipeline:
    """
    Tiered Vision Processing Pipeline.
    
    Automatically picks the cheapest strategy:
      1. Intent detection (free, instant)
      2. PaddleOCR for text (free, ~1-2s, falls back to Tesseract)
      3. Ollama/Moondream for objects (free, ~2-4s)
      4. Cloud fallback (paid, full quality)
    """

    def __init__(self):
        self.ollama = OllamaVision()
        self._ocr_available = None

    async def _check_ocr(self) -> bool:
        """Check if PaddleOCR or Tesseract is available."""
        if self._ocr_available is not None:
            return self._ocr_available
        try:
            model = _get_paddle_ocr()
            if model:
                self._ocr_available = True
                log.info("PaddleOCR available")
                return True
        except Exception:
            pass
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._ocr_available = True
            log.info("Tesseract OCR available (fallback)")
        except Exception:
            self._ocr_available = False
            log.info("No OCR engine available")
        return self._ocr_available

    async def process(self, image_b64: str, query: str) -> dict:
        """
        Main entry point: process an image with the cheapest viable strategy.
        
        Returns:
            {
                "strategy": "ocr_only" | "local_vision" | "augmented" | "passthrough",
                "text_description": str | None,   # Text to substitute for image
                "ocr_text": str | None,            # Raw OCR output
                "vision_description": str | None,  # Local vision output
                "intent": VisionIntent,
                "can_skip_image": bool,            # True = don't send image to LLM
                "preprocessing": dict,             # Image resize info
            }
        """
        intent = detect_vision_intent(query)
        log.info(f"Vision intent: {intent.value} for query: '{query[:60]}'")

        result = {
            "strategy": "passthrough",
            "text_description": None,
            "ocr_text": None,
            "vision_description": None,
            "intent": intent,
            "can_skip_image": False,
            "preprocessing": {},
        }

        # ─── Strategy 1: OCR-focused (documents, text) ───────────────
        if intent == VisionIntent.TEXT_EXTRACT:
            if await self._check_ocr():
                ocr = await paddle_ocr(image_b64)
                result["ocr_text"] = ocr.get("text", "")

                if ocr["success"] and len(ocr["text"]) > 20:
                    # OCR found enough text → skip image entirely
                    result["strategy"] = "ocr_only"
                    result["text_description"] = (
                        f"[Extracted text from image via OCR]\n"
                        f"{ocr['text']}"
                    )
                    result["can_skip_image"] = True
                    log.info(f"OCR sufficient: {len(ocr['text'])} chars, "
                             f"skipping image → text-only")
                    return result

            # OCR failed/unavailable → try local vision
            log.info("OCR insufficient, trying local vision for text extraction")

        # ─── Strategy 2: Simple object/scene (local vision) ──────────
        if intent in (VisionIntent.OBJECT_SIMPLE, VisionIntent.TEXT_EXTRACT,
                       VisionIntent.CODE_SCREENSHOT):
            ollama_result = await self.ollama.analyze(image_b64, query)

            if ollama_result["success"]:
                desc = ollama_result["description"]
                result["vision_description"] = desc

                # If description is detailed enough, skip image
                if len(desc) > 50:
                    result["strategy"] = "local_vision"
                    result["text_description"] = (
                        f"[Image analysis by local model ({ollama_result['model']})]\n"
                        f"{desc}"
                    )
                    result["can_skip_image"] = True
                    log.info(f"Local vision sufficient: {len(desc)} chars, "
                             f"skipping image → text-only")
                    return result
                else:
                    # Partial info — augment but still send image
                    result["strategy"] = "augmented"
                    result["text_description"] = (
                        f"[Preliminary image analysis: {desc}]\n"
                        f"Please analyze the image in more detail."
                    )
                    log.info(f"Local vision partial: {len(desc)} chars, "
                             f"augmenting with image")
                    return result

        # ─── Strategy 3: Complex vision → passthrough to cloud ───────
        # Preprocess image for size optimization before cloud send
        processed_b64, prep_info = preprocess_image(
            image_b64, max_pixels=2048, quality=85
        )
        result["preprocessing"] = prep_info
        result["strategy"] = "passthrough"
        log.info(f"Complex vision: passthrough to cloud model "
                 f"(savings: {prep_info.get('savings_pct', 0)}%)")

        return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: Extract base64 image data from ChatMessage content
# ═══════════════════════════════════════════════════════════════════════════════

def extract_image_b64(content: list) -> Optional[str]:
    """Extract the first base64 image from a multimodal content list."""
    if not isinstance(content, list):
        return None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image_url":
            url = item.get("image_url", {}).get("url", "")
            if url.startswith("data:") and "," in url:
                return url.split(",", 1)[1]
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                return source.get("data", "")
    return None


def build_text_only_message(original_content: list, text_description: str) -> str:
    """
    Replace image in multimodal content with text description.
    Returns a plain text string suitable for cheap models.
    """
    parts = []
    if isinstance(original_content, list):
        for item in original_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
    
    if text_description:
        parts.insert(0, text_description)
    
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level pipeline instance
# ═══════════════════════════════════════════════════════════════════════════════

_pipeline: Optional[VisionPipeline] = None


def get_vision_pipeline() -> VisionPipeline:
    """Get or create the global VisionPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VisionPipeline()
    return _pipeline
