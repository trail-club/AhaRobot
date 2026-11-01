"""Proper-noun segmentation: a VLM proposes phrases, SAM 3.1 proposes masks, the VLM picks.

SAM 3.1 matches appearance, not brands or packaging text, so a prompt such as
"pringles" also returns look-alike cans. The VLM looks at the image to turn the
user's prompt into generic phrases, then reads the numbered candidates and keeps
only the ones that are the target. An instance turned away from the camera has
no readable logo, so each rejected candidate is then compared side by side with
a kept one. Masks always come from SAM 3.1.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import re
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .sam31 import color_hex, render_candidates, validate_rgb, validate_threshold


# Qwen3.8 on the DGX Spark (tools/local-llm), reached over Cloudflare One (WARP).
DEFAULT_VLM_URL = "http://10.99.0.1:8080/v1"
DEFAULT_VLM_MODEL = "qwen3.8-27b"
DEFAULT_CANDIDATE_THRESHOLD = 0.3
MAX_PHRASES = 3
MAX_CANDIDATES = 8
# Candidates from different phrases that overlap this much are the same object.
DUPLICATE_IOU = 0.8
# Keep result.json small; the end of the reasoning holds the decision.
MAX_THINKING_CHARS = 4000
# Side-by-side crops for the same-product check.
MATCH_PAD_PX = 6
MATCH_HEIGHT_PX = 360
MATCH_GAP_PX = 20

PHRASE_PROMPT = """The user wants to segment: "{target}". It may be a product or brand name in any language.
Look at the image. Give up to {count} short generic English noun phrases (2-4 words, no brand names, no text printed on packaging) describing what the target looks like in this image, so an open-vocabulary segmentation model can find its object category. Examples: "red snack can", "paper snack cup".
Reply with JSON only: {{"phrases": ["..."]}}"""

# Image 1 is the photo, image 2 the numbered candidates: hatching hides logos,
# and the colour drawn over a candidate changes how its packaging looks.
SELECT_PROMPT = """Image 1 is the original photo. Image 2 is the same photo with candidate masks drawn with diagonal hatching; each candidate has its own colour and a number tag (1-{count}) at its top-left corner.
Target: "{target}". It may be a product or brand name. The hatching hides details, so read packaging text, logos and colours in image 1, and use image 2 only to see which region each number covers.
The target may appear more than once (for example two identical products), and an instance may be turned so that its logo or name is not visible (back or side, nutrition label, partly out of frame or hidden). Check every candidate one by one. Keep a candidate if its packaging shows the target, or if it has the same shape, size, colours and packaging design as a candidate that does. Do not keep a different product only because it has a similar colour or shape category.
Use an empty list if none is.
Reply with JSON only: {{"keep": [numbers]}}"""

MATCH_PROMPT = """Image 1 shows object A (left) and object B (right), cropped from the same photo and drawn at the same real-world scale (white is padding).
A is a "{target}". Size measured with the depth camera: A is {size_a}; B is {size_b}.
Objects may be turned, so B may show its back or side instead of the logo, or be partly cut off (then it looks smaller than it is).
Is B the same product as A (same brand and product line)? Items of one product line have the same size and proportions, so a clearly different size means a different product unless B is cut off. Different brands with a similar colour or shape are not the same.
Reply with JSON only: {{"same": true or false}}"""

KIND_PROMPT = """The photo shows one object placed near the centre. What kind of object is it, by its shape and material? Answer with a short generic English noun phrase of 1-3 words, without any brand or product name and without its contents (examples: "cylindrical can", "plastic bottle", "paper cup", "cardboard box"; for unpackaged items name the item, e.g. "banana").
Reply with JSON only: {"phrase": "..."}"""


class VlmClient:
    """Minimal OpenAI-compatible chat client for one instruction about some images.

    The Qwen3.8 server takes images as data URLs (at most 4 per request).
    Reasoning is off by default: on the candidate selection it took minutes and
    thousands of tokens and still chose worse (docs/context/perception.md).
    With `thinking=True` the reasoning is kept in `last_thinking`.
    """

    def __init__(
        self, url=DEFAULT_VLM_URL, model=DEFAULT_VLM_MODEL, timeout=180, thinking=False
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.thinking = thinking
        self.last_thinking = ""

    def ask(self, prompt: str, *images: np.ndarray) -> str:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            buffer = BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # Reasoning counts towards max_tokens; leave room for the answer.
            "max_tokens": 4096,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        request = urllib.request.Request(
            self.url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"VLM request failed: HTTP {exc.code} {exc.read()[:200]!r}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"VLM is not reachable at {self.url}: {exc}") from exc
        choices = reply.get("choices") or [{}]
        message = choices[0].get("message") or {}
        self.last_thinking = message.get("reasoning_content") or ""
        answer = message.get("content") or ""
        if not answer.strip():
            raise RuntimeError(
                "VLM returned no answer (reasoning used the token budget)"
            )
        return answer


class _RecordingClient:
    """Record each VLM exchange (answer, reasoning, time) for the flow report.

    Set `name` before a call to name the step it records.
    """

    def __init__(self, client):
        self.client = client
        self.steps = []
        self.name = None

    def ask(self, prompt, *images):
        started = time.perf_counter()
        try:
            answer = self.client.ask(prompt, *images)
        finally:
            thinking = getattr(self.client, "last_thinking", "") or ""
            self.steps.append(
                {
                    "name": self.name,
                    "seconds": time.perf_counter() - started,
                    "thinking": thinking[-MAX_THINKING_CHARS:],
                }
            )
        self.steps[-1]["answer"] = answer
        return answer


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def parse_json_object(text: str) -> dict:
    """Read the JSON object in a reply that may be wrapped in a code fence."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VLM reply has no JSON object: {text[:200]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM reply is not valid JSON: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError("VLM reply must be a JSON object")
    return data


def kind_phrase(client, rgb: np.ndarray) -> str:
    """Generic SAM phrase for the object at the centre of a reference shot.

    Asked once per product so that later detection needs only the product's
    reference images, not a VLM.
    """
    h, w = rgb.shape[:2]
    centre = np.ascontiguousarray(rgb[h // 5 : h - h // 5, w // 5 : w - w // 5])
    phrase = str(parse_json_object(client.ask(KIND_PROMPT, centre)).get("phrase", ""))
    phrase = " ".join(phrase.lower().split())
    if not re.fullmatch(r"[a-z][a-z0-9 '&.-]{0,40}", phrase):
        raise ValueError(f"VLM gave no usable kind phrase: {phrase!r}")
    return phrase


def propose_phrases(client, rgb: np.ndarray, prompt: str) -> list[str]:
    reply = parse_json_object(
        client.ask(PHRASE_PROMPT.format(target=prompt, count=MAX_PHRASES), rgb)
    )
    proposed = reply.get("phrases")
    if not isinstance(proposed, list):
        raise ValueError("VLM reply must contain a phrases list")
    # SAM 3.1 may already know an English name, so try the prompt itself too.
    candidates = [prompt] if re.fullmatch(r"[A-Za-z][A-Za-z0-9 '&.-]*", prompt) else []
    candidates += [p for p in proposed if isinstance(p, str)][:MAX_PHRASES]
    phrases = []
    for phrase in candidates:
        phrase = " ".join(phrase.lower().split())
        if phrase and len(phrase) <= 60 and phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        raise ValueError("VLM proposed no usable phrases")
    return phrases


def merge_candidates(results, height: int, width: int):
    """Pool masks from several phrases, highest score first, without duplicates.

    Also returns, per phrase, what happened to each SAM mask: it became
    candidate N, duplicated candidate N, or exceeded MAX_CANDIDATES.
    """
    pooled, runs = [], []
    for phrase, outputs in results:
        masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
        scores = np.asarray(outputs["out_probs"], dtype=np.float32)
        if scores.ndim != 1 or masks.shape != (len(scores), height, width):
            raise ValueError("SAM 3.1 masks/scores have unexpected dimensions")
        run = {"phrase": phrase, "masks": []}
        for score, mask in zip(scores, masks):
            if mask.any():
                entry = {
                    "score": float(score),
                    "bbox_xyxy": _bbox(mask),
                    "pixels": int(mask.sum()),
                }
                run["masks"].append(entry)
                pooled.append((float(score), mask, phrase, entry))
        runs.append(run)
    pooled.sort(key=lambda item: -item[0])
    kept = []
    for score, mask, phrase, entry in pooled:
        duplicate = next(
            (
                number
                for number, (_, other, _) in enumerate(kept, 1)
                if (mask & other).sum() / (mask | other).sum() > DUPLICATE_IOU
            ),
            None,
        )
        if duplicate is not None:
            entry["outcome"] = f"duplicate of {duplicate}"
        elif len(kept) == MAX_CANDIDATES:
            entry["outcome"] = "over limit"
        else:
            kept.append((score, mask, phrase))
            entry["outcome"] = f"candidate {len(kept)}"
    masks = np.array([m for _, m, _ in kept], dtype=bool).reshape(-1, height, width)
    scores = np.array([s for s, _, _ in kept], dtype=np.float32)
    return masks, scores, [p for _, _, p in kept], runs


def select_candidates(
    client, rgb: np.ndarray, image: np.ndarray, prompt: str, count: int
) -> list[int]:
    reply = parse_json_object(
        client.ask(SELECT_PROMPT.format(target=prompt, count=count), rgb, image)
    )
    keep = reply.get("keep")
    if not isinstance(keep, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) for n in keep
    ):
        raise ValueError("VLM reply must contain a keep list of integers")
    if any(not 1 <= n <= count for n in keep):
        raise ValueError(f"VLM kept candidates outside 1..{count}: {keep}")
    return sorted(set(keep))


def focal_lengths(camera_info) -> tuple[float, float] | None:
    """(fx, fy) in pixels from a CameraInfo-like dict, or None if unusable."""
    k = (camera_info or {}).get("k") or []
    return (k[0], k[4]) if len(k) == 9 and k[0] > 0 and k[4] > 0 else None


def physical_size(mask, depth_m, intrinsics):
    """Box width/height in cm at the mask's median depth, or None without depth.

    `cut_off` marks a box touching the image edge: the object may be larger.
    """
    if depth_m is None or intrinsics is None:
        return None
    samples = depth_m[mask]
    valid = samples[np.isfinite(samples) & (samples > 0)]
    if not len(valid):
        return None
    depth = float(np.median(valid))
    fx, fy = intrinsics
    x0, y0, x1, y1 = _bbox(mask)
    height, width = mask.shape
    return {
        "width_cm": round((x1 - x0) * depth / fx * 100, 1),
        "height_cm": round((y1 - y0) * depth / fy * 100, 1),
        "depth_m": round(depth, 3),
        "cut_off": x0 == 0 or y0 == 0 or x1 == width or y1 == height,
    }


def size_text(size) -> str:
    if size is None:
        return "of unknown size"
    text = f"about {size['height_cm']:.0f} cm tall and {size['width_cm']:.0f} cm wide"
    return text + (" (cut off by the image edge)" if size["cut_off"] else "")


def pair_image(
    rgb: np.ndarray, reference: np.ndarray, other: np.ndarray, depths=(None, None)
) -> np.ndarray:
    """Crop two masks' boxes and place them side by side (A | B) at one scale.

    With depths, pixels are scaled by distance so that both crops share a
    real-world scale; otherwise they share the photo's pixel scale. Either way
    a short can stays short next to a tall one.
    """
    crops, scales = [], []
    for mask, depth in zip((reference, other), depths):
        x0, y0, x1, y1 = _bbox(mask)
        crops.append(
            Image.fromarray(
                rgb[
                    max(0, y0 - MATCH_PAD_PX) : y1 + MATCH_PAD_PX,
                    max(0, x0 - MATCH_PAD_PX) : x1 + MATCH_PAD_PX,
                ]
            )
        )
        scales.append(depth if depth and None not in depths else 1.0)
    factor = MATCH_HEIGHT_PX / max(c.height * k for c, k in zip(crops, scales))
    crops = [
        c.resize(
            (max(1, round(c.width * k * factor)), max(1, round(c.height * k * factor)))
        )
        for c, k in zip(crops, scales)
    ]
    canvas = Image.new(
        "RGB",
        (crops[0].width + MATCH_GAP_PX + crops[1].width, MATCH_HEIGHT_PX),
        (255, 255, 255),
    )
    # Bottom-aligned, like objects standing on one table.
    canvas.paste(crops[0], (0, MATCH_HEIGHT_PX - crops[0].height))
    canvas.paste(
        crops[1], (crops[0].width + MATCH_GAP_PX, MATCH_HEIGHT_PX - crops[1].height)
    )
    return np.asarray(canvas)


def match_candidate(client, image: np.ndarray, prompt: str, sizes=(None, None)) -> bool:
    reply = parse_json_object(
        client.ask(
            MATCH_PROMPT.format(
                target=prompt, size_a=size_text(sizes[0]), size_b=size_text(sizes[1])
            ),
            image,
        )
    )
    same = reply.get("same")
    if not isinstance(same, bool):
        raise ValueError("VLM reply must contain a same boolean")
    return same


def _match_rejected(client, rgb, masks, selected, prompt, depth_m, intrinsics, images):
    """Compare each rejected candidate with the top selected one; return the checks.

    Each side-by-side image is added to `images` as `match_N.png`.
    """
    # The highest-scoring chosen candidate is the reference appearance.
    reference = selected[0]
    reference_size = physical_size(masks[reference - 1], depth_m, intrinsics)
    matches = []
    for number in range(1, len(masks) + 1):
        if number in selected:
            continue
        size = physical_size(masks[number - 1], depth_m, intrinsics)
        sizes = (reference_size, size)
        pair = pair_image(
            rgb,
            masks[reference - 1],
            masks[number - 1],
            tuple(s and s["depth_m"] for s in sizes),
        )
        client.name = f"match {number}"
        same = match_candidate(client, pair, prompt, sizes)
        images[f"match_{number}.png"] = pair
        matches.append(
            {
                "number": number,
                "reference": reference,
                "same": same,
                "reference_size": reference_size,
                "size": size,
            }
        )
    return matches


def segment_with_vlm(
    segmenter,
    client,
    rgb,
    prompt: str,
    candidate_threshold: float,
    match=True,
    depth_m=None,
    intrinsics=None,
):
    """Return SAM outputs for the kept candidates, the VLM record and its images.

    Images are named result files: the numbered candidates and, per compared
    candidate N, the side-by-side crops `match_N.png`. Aligned depth (meters)
    and (fx, fy) give the size comparison real-world units.
    """
    validate_rgb(rgb)
    validate_threshold(candidate_threshold)
    height, width = rgb.shape[:2]
    recording = _RecordingClient(client)
    recording.name = "phrases"
    phrases = propose_phrases(recording, rgb, prompt)
    results, sam_seconds = [], []
    for phrase in phrases:
        segmenter.prompt = phrase
        started = time.perf_counter()
        results.append((phrase, segmenter.predict(rgb, threshold=candidate_threshold)))
        sam_seconds.append(time.perf_counter() - started)
    masks, scores, sources, runs = merge_candidates(results, height, width)
    for run, seconds in zip(runs, sam_seconds):
        run["seconds"] = seconds
    selected, matches = [], []
    images = {"candidates.png": rgb.copy()}
    if len(masks):
        images["candidates.png"] = render_candidates(rgb, masks)
        recording.name = "select"
        selected = select_candidates(
            recording, rgb, images["candidates.png"], prompt, len(masks)
        )
    if match and selected:
        matches = _match_rejected(
            recording, rgb, masks, selected, prompt, depth_m, intrinsics, images
        )
    keep = sorted(selected + [m["number"] for m in matches if m["same"]])
    kept = [n - 1 for n in keep]
    outputs = {"out_binary_masks": masks[kept], "out_probs": scores[kept]}

    def via(number):
        if number in selected:
            return "select"
        return "match" if number in keep else None

    record = {
        "model": client.model,
        "phrases": phrases,
        "candidate_threshold": candidate_threshold,
        "candidates": [
            {
                "number": i + 1,
                "phrase": sources[i],
                "score": float(scores[i]),
                "bbox_xyxy": _bbox(masks[i]),
                "kept": i + 1 in keep,
                "via": via(i + 1),
                "color": color_hex(i + 1),
            }
            for i in range(len(masks))
        ],
        "selected": selected,
        "matches": matches,
        "keep": keep,
        "vlm_seconds": sum(step["seconds"] for step in recording.steps),
        "steps": recording.steps,
        "sam_runs": runs,
    }
    return outputs, record, images
