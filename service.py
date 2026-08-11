"""
Thin FastAPI wrapper around Seed-VC v2 (hubert-bsqvae-small: cfm_small.pth +
ar_base.pth), run in this folder's own Python 3.11 venv (seed-vc/venv/) -
needs torch 2.4.0+cu124, a much newer stack than backend_dlc's own
fastapi/aiortc environment, so it cannot run in-process. backend_dlc's
voice/client.py talks to this over plain local HTTP - same reasoning, and
the exact same request/response contract, as chatterbox/service.py (which
backend/, the original DeepFaceLive app, still uses unchanged).

Real per-chunk latency was benchmarked on this box's actual L4 GPU before
this was built (see PROJECT_HANDOFF.md): 30 diffusion steps (the reference
CLI's own real default, inference_v2.py) averaged 1.31-1.71s for a 2.0s
chunk (RTF 0.65-0.86x across two separate measurement passes), 10 steps
averaged 0.62-1.0s (RTF 0.31-0.5x) - both technically fit a live 2s cadence,
despite the upstream repo documenting no real-time support for v2 at all.
Defaults to 30 (not 10) as of 2026-08-08: 10 was an untested latency-only
guess that produced audibly bad quality in real live use - see
PROJECT_HANDOFF.md section 27 for the full story. 30 leaves less latency
margin for when the video swap model is also running concurrently on this
same GPU (untested combination) - override via --diffusion-steps if a
future concurrent-load test says this needs to come back down.

Endpoints:
    GET  /health              -> {"status": "ok", "loaded": bool}
    POST /set-target-voice    -> body {"path": "default" | "<absolute .wav path>"}
    POST /convert              -> raw 16kHz mono s16 PCM in the request body,
                                   raw 22050Hz mono s16 PCM in the response body
                                   (Seed-VC v2's native vocoder rate - unlike
                                   chatterbox's fixed 24kHz output, so
                                   backend_dlc/webrtc/tracks.py's _VC_OUTPUT_SR
                                   was changed to match rather than resampling
                                   here for no reason).
"""
import argparse
import logging
import sys
import tempfile
import threading
import time
import wave
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

SEEDVC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEEDVC_DIR))  # os.chdir alone doesn't put this on sys.path - modules.v2.* imports need it

# convert_voice_with_streaming()'s stream_output=False path is dead code in
# the upstream repo (confirmed by reading modules/v2/vc_wrapper.py directly):
# _stream_wave_chunks() computes the final audio correctly either way, but
# the generator's `yield mp3_bytes, full_audio` is gated behind
# `if stream_output`, so with False nothing is ever yielded and the result is
# unreachable - despite the docstring's claim it "returns the full audio as
# a numpy array". stream_output=True is the only code path that actually
# works, which means every call pays for an mp3-encode via pydub -> ffmpeg,
# even though we only want the raw audio. ffmpeg isn't on PATH in this venv,
# so bundle the copy already proven working for Deep-Live-Cam (same gap, same
# fix) and point pydub at it directly rather than fighting PATH.
import pydub
pydub.AudioSegment.converter = str(SEEDVC_DIR / "ffmpeg.exe")
pydub.AudioSegment.ffprobe = str(SEEDVC_DIR / "ffprobe.exe")

# _run_conversion() below always discards the mp3_bytes half of each yielded
# tuple - we only ever want full_audio. But _stream_wave_chunks() unconditionally
# calls AudioSegment(...).export(format="mp3", ...) whenever stream_output=True,
# which shells out to the ffmpeg.exe above on every single chunk. Measured
# this real cost end-to-end via HTTP (not assumed): ~2.4s per 2.0s chunk with
# it, vs ~0.6s without (matching the earlier in-process benchmark) - the
# subprocess spawn + encode alone blew straight past the live 2s chunk
# budget for output nothing ever reads. Stubbing .export() out entirely
# rather than resampling/optimizing it, since it's fully unused dead weight
# for this service's purposes.
class _DiscardedExport:
    def read(self):
        return b""


def _skip_unused_mp3_export(self, *args, **kwargs):
    return _DiscardedExport()


pydub.AudioSegment.export = _skip_unused_mp3_export

# Bundled outside this clone (in userdata-dlc/, not seed-vc/) so the seed-vc
# checkout itself stays an unmodified upstream clone - same reasoning as why
# Deep-Live-Cam's temporary default face used to live under userdata-dlc/
# rather than inside the Deep-Live-Cam/ submodule.
_DEFAULT_VOICE_PATH = SEEDVC_DIR.parent / "userdata-dlc" / "default_voice" / "default_voice.wav"

_INPUT_SR = 16000  # matches backend_dlc's _VC_INPUT_SR and how uploaded voice clips are stored

# Real finding, not a guess: convert_voice_with_streaming() concatenates the
# FULL target reference clip into every single diffusion step's attention
# context (self.cfm_length_regulator's prompt_condition scales directly with
# target length) - measured a full ~5.6x slowdown (5.5-5.9it/s vs 32-36it/s
# at the same diffusion_steps) going from a 5s reference clip to a 37s one.
# backend_dlc's /api/voices/upload has no duration limit today (20MB is a
# generous *file size* cap, not a duration one), so this is enforced here,
# once per voice selection rather than trusting every caller to send short
# clips - not just a fix for the bundled default voice.
_MAX_TARGET_SECONDS = 6.0  # kept at 6.0 after a real sweep, not an arbitrary choice
                           # (real user report: converted voice didn't sound like the
                           # target and had glitches within a chunk - tried raising
                           # this to see if it helped). Measured RTF at 30 diffusion
                           # steps: 6s->0.77-0.9x, 8s->0.89-0.96x (razor-thin margin),
                           # 10s->1.1-1.2x (already too slow), 15s->1.6-1.7x
                           # (unusable). Reverted to 6.0 rather than keep 8.0, since
                           # 8.0's margin would likely vanish entirely under the
                           # still-untested concurrent-GPU-load case (video swap
                           # running at the same time) - dropped chunks (audible
                           # gaps) are worse than the current quality issue, and
                           # duration alone wasn't fixing it anyway. See
                           # PROJECT_HANDOFF.md section 29 - the quality issue needs
                           # a different diagnosis, this lever is maxed out.
_TARGET_CACHE_PATH = Path(tempfile.gettempdir()) / "seedvc_target_trimmed.wav"

# Logs to both the console (already visible in the "SwapX DLC - Voice" window)
# and userdata-dlc/logs/voice_service.log - same directory backend_dlc's own
# logger already writes backend.log to, so there's one place to look for
# voice activity, not two. This is what actually lets per-chunk latency be
# monitored during a real session instead of only measured via one-off manual
# curl timing tests, per the user's explicit ask - the earlier latency
# numbers in this file's docstring were all gathered that way, by hand.
_LOG_DIR = SEEDVC_DIR.parent / "userdata-dlc" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("seedvc")
logger.setLevel(logging.INFO)
_log_fmt = logging.Formatter("%(asctime)s %(levelname)s [seedvc] %(message)s")
_file_handler = RotatingFileHandler(_LOG_DIR / "voice_service.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
logger.addHandler(_file_handler)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)
logger.addHandler(_console_handler)

app = FastAPI(title="Seed-VC v2 Voice Service")

_vc_wrapper = None
_device: Optional[torch.device] = None
_dtype = torch.float16
_current_target_path: str = str(_DEFAULT_VOICE_PATH)
_diffusion_steps = 30  # matches inference_v2.py's own reference CLI default - see module
                       # docstring: the earlier default of 10 was a latency-only guess that
                       # was never quality-tested and produced audibly bad output in real use
                       # (real user feedback, live-tested through the actual app) - 30 still
                       # measured comfortably inside the 2.0s chunk budget on this box's L4
                       # (~1.5-1.7s/chunk, RTF 0.77-0.86x), just with less margin than 10
                       # steps had. Overridable via --diffusion-steps.
_lock = threading.Lock()  # mirrors chatterbox/service.py - inference isn't safe for concurrent calls


@app.on_event("startup")
def on_startup():
    global _vc_wrapper, _device, _current_target_path

    if not _DEFAULT_VOICE_PATH.exists():
        raise RuntimeError(
            f"Default voice reference clip missing: {_DEFAULT_VOICE_PATH} - "
            "copy a reference .wav there before starting this service."
        )

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")  # would work but be far too slow for live use - not expected on this box

    from hydra.utils import instantiate
    from omegaconf import DictConfig

    import os
    os.chdir(SEEDVC_DIR)  # configs/v2/vc_wrapper.yaml is loaded via a path relative to cwd

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    # None/None -> auto-downloads cfm_small.pth/ar_base.pth from the
    # Plachta/Seed-VC HF repo on first run (confirmed - see PROJECT_HANDOFF.md).
    vc_wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
    vc_wrapper.to(_device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=_dtype, device=_device)
    _vc_wrapper = vc_wrapper

    # Route the initial default through the same trim-to-cache path
    # /set-target-voice uses, rather than trusting the bundled file to
    # already be short (belt and suspenders - see _MAX_TARGET_SECONDS above).
    _current_target_path = _prepare_target_voice(str(_DEFAULT_VOICE_PATH))

    # Warm-up: pays for CUDA kernel compilation once here instead of on
    # whichever real user's first chunk hits this process first - mirrors
    # chatterbox/service.py's own startup warm-up, including using a clip of
    # the real pipeline's actual chunk duration (2.0s), not an arbitrary
    # shorter one - confirmed empirically for chatterbox that a shorter
    # warm-up clip does NOT speed up subsequent real-length chunks.
    _WARMUP_SECONDS = 2.0
    with tempfile.TemporaryDirectory() as tmpdir:
        warmup_path = Path(tmpdir) / "warmup.wav"
        with wave.open(str(warmup_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_INPUT_SR)
            w.writeframes(b"\x00\x00" * int(_WARMUP_SECONDS * _INPUT_SR))
        try:
            _run_conversion(str(warmup_path), _current_target_path)
        except Exception:
            pass  # non-fatal - worst case the first real chunk pays the cost instead


def _run_conversion(source_path: str, target_path: str) -> tuple[int, np.ndarray]:
    generator = _vc_wrapper.convert_voice_with_streaming(
        source_audio_path=source_path,
        target_audio_path=target_path,
        diffusion_steps=_diffusion_steps,
        length_adjust=1.0,
        intelligebility_cfg_rate=0.7,
        similarity_cfg_rate=0.7,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.0,
        convert_style=False,
        anonymization_only=False,
        device=_device,
        dtype=_dtype,
        # Must be True - stream_output=False's advertised "just returns the
        # array" behavior is dead code upstream (see the sys.path comment
        # above for why). We only care about the final tuple's full_audio,
        # not the per-chunk mp3_bytes.
        stream_output=True,
    )
    full_audio = None
    for output in generator:
        _, full_audio = output
    if full_audio is None:
        raise RuntimeError("Seed-VC v2 returned no audio")
    sr, audio = full_audio
    return sr, audio


def _prepare_target_voice(source_path: str) -> str:
    """
    Trims a reference clip to _MAX_TARGET_SECONDS and writes it to a fixed
    cache path, returning that path. Runs once per voice selection (not per
    chunk) - see _MAX_TARGET_SECONDS' comment for why this exists at all.
    Clips already under the cap are still re-written (simpler than branching,
    and the file is small so the cost is negligible either way).
    """
    data, sr = sf.read(source_path, dtype="float32")
    max_samples = int(_MAX_TARGET_SECONDS * sr)
    if data.shape[0] > max_samples:
        data = data[:max_samples]
    sf.write(str(_TARGET_CACHE_PATH), data, sr)
    return str(_TARGET_CACHE_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _vc_wrapper is not None}


class SetVoiceBody(BaseModel):
    path: str


@app.post("/set-target-voice")
def set_target_voice(body: SetVoiceBody):
    global _current_target_path
    if _vc_wrapper is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    with _lock:
        if body.path == "default":
            _current_target_path = _prepare_target_voice(str(_DEFAULT_VOICE_PATH))
            return {"ok": True}

        p = Path(body.path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"voice file not found: {body.path}")
        _current_target_path = _prepare_target_voice(str(p))
        return {"ok": True}


@app.post("/convert")
async def convert(request: Request):
    if _vc_wrapper is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="empty body")

    t0 = time.monotonic()
    chunk_seconds = len(pcm) / 2 / _INPUT_SR  # s16 mono = 2 bytes/sample

    # Temp-file round trip, not an in-memory buffer - convert_voice_with_streaming
    # takes a path (calls librosa.load(path, sr=self.sr) internally), same
    # reasoning as chatterbox/service.py's identical pattern.
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.wav"
        with wave.open(str(in_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # s16
            w.setframerate(_INPUT_SR)
            w.writeframes(pcm)

        with _lock:
            try:
                sr, audio = _run_conversion(str(in_path), _current_target_path)
            except Exception:
                elapsed = time.monotonic() - t0
                logger.exception("convert failed after %.3fs (chunk=%.2fs)", elapsed, chunk_seconds)
                raise

    out_np = np.clip(audio, -1.0, 1.0)
    out_pcm = (out_np * 32767.0).astype(np.int16).tobytes()

    elapsed = time.monotonic() - t0
    rms = float(np.sqrt(np.mean(out_np ** 2))) if out_np.size else 0.0
    # Pure model+I/O time on this side (no HTTP/network hop) - compare against
    # backend_dlc/logs/backend.log's "voice chunk converted" lines, which
    # include the HTTP round-trip, to see how much of the total is network vs
    # actual conversion. RMS logged too since silent/near-zero output would
    # look like a "quality" complaint but is actually a different bug.
    logger.info(
        "convert: %.3fs for a %.2fs chunk (rtf=%.2fx), diffusion_steps=%d, out_rms=%.3f",
        elapsed, chunk_seconds, elapsed / chunk_seconds if chunk_seconds else 0.0, _diffusion_steps, rms,
    )

    return Response(content=out_pcm, media_type="application/octet-stream", headers={"X-Sample-Rate": str(sr)})


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    args = parser.parse_args()

    _diffusion_steps = args.diffusion_steps
    uvicorn.run(app, host=args.host, port=args.port)
