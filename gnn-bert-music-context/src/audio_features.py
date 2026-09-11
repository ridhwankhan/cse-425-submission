"""Audio feature extraction: log-mel, chroma, MFCC, segmentation (torchaudio-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


def load_audio(
    path: Union[str, Path],
    sr: int = 22050,
    mono: bool = True,
    duration: float = 20.0,
) -> Tuple[np.ndarray, int]:
    path = Path(path)
    # 1) torchaudio (handles mp3 via torchcodec/ffmpeg backends when available)
    try:
        import torch
        import torchaudio

        wav, file_sr = torchaudio.load(str(path))
        if mono and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, int(file_sr), sr)
        y = wav.squeeze(0).numpy().astype(np.float32)
        if duration and duration > 0:
            y = y[: int(sr * duration)]
        if y.size == 0:
            raise RuntimeError("empty audio")
        return y, sr
    except Exception:
        pass

    # 2) ffmpeg decode to wav bytes then soundfile/wave
    try:
        import shutil
        import subprocess
        import tempfile
        import wave

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
            for p in winget.rglob("ffmpeg.exe"):
                ffmpeg = str(p)
                break
        if ffmpeg:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-t",
                str(duration),
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            with wave.open(str(tmp_path), "rb") as w:
                frames = w.readframes(w.getnframes())
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            tmp_path.unlink(missing_ok=True)
            if y.size:
                return y, sr
    except Exception:
        pass

    # 3) soundfile for wav/flac
    try:
        import soundfile as sf

        y, file_sr = sf.read(str(path), always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if file_sr != sr:
            duration_s = len(y) / float(file_sr)
            new_len = int(duration_s * sr)
            x_old = np.linspace(0, 1, num=len(y), endpoint=False)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            y = np.interp(x_new, x_old, y).astype(np.float32)
        if duration and duration > 0:
            y = y[: int(sr * duration)]
        return y, sr
    except Exception:
        return np.zeros(sr * 10, dtype=np.float32), sr


def normalize_track(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(y)) + eps
    return (y / peak).astype(np.float32)


def _stft_mag(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop_length
    if n_frames <= 0:
        n_frames = 1
    out = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size))
        spec = np.fft.rfft(frame * window)
        out[:, i] = np.abs(spec).astype(np.float32)
    return out


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fmin, fmax = 0.0, sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (j - left) / max(1, center - left)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i, j] = (right - j) / max(1, right - center)
    return fb


def log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    try:
        import torch
        import torchaudio

        wav = torch.from_numpy(y).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
        )(wav)
        log_S = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
        return log_S.squeeze(0).numpy().astype(np.float32)
    except Exception:
        mag = _stft_mag(y, n_fft=n_fft, hop_length=hop_length)
        fb = _mel_filterbank(sr, n_fft, n_mels)
        mel = fb @ (mag ** 2)
        log_S = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_S = log_S - np.max(log_S)
        return log_S.astype(np.float32)


def chroma_features(
    y: np.ndarray,
    sr: int = 22050,
    n_chroma: int = 12,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    mag = _stft_mag(y, n_fft=n_fft, hop_length=hop_length)
    freqs = np.linspace(0, sr / 2.0, mag.shape[0])
    chroma = np.zeros((n_chroma, mag.shape[1]), dtype=np.float32)
    for i, f in enumerate(freqs):
        if f < 1.0:
            continue
        # MIDI pitch class
        midi = 69 + 12 * np.log2(f / 440.0)
        pc = int(np.round(midi)) % 12
        chroma[pc] += mag[i]
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8
    return (chroma / norms).astype(np.float32)


def mfcc_features(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    hop_length: int = 512,
) -> np.ndarray:
    try:
        import torch
        import torchaudio

        wav = torch.from_numpy(y).unsqueeze(0)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr, n_mfcc=n_mfcc, melkwargs={"hop_length": hop_length, "n_mels": 128}
        )(wav)
        return mfcc.squeeze(0).numpy().astype(np.float32)
    except Exception:
        mel = log_mel_spectrogram(y, sr=sr, hop_length=hop_length)
        # DCT-II
        n_mels, t = mel.shape
        n = np.arange(n_mels)[:, None]
        k = np.arange(n_mfcc)[None, :]
        basis = np.cos(np.pi * (n + 0.5) * k / n_mels).astype(np.float32)
        return (basis.T @ mel).astype(np.float32)


def frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    return (np.arange(n_frames) * hop_length / float(sr)).astype(np.float32)


def segment_indices(
    duration_sec: float,
    segment_seconds: float = 5.0,
    hop_seconds: float = 2.5,
) -> np.ndarray:
    if duration_sec <= 0:
        return np.array([0.0], dtype=np.float32)
    starts = []
    t = 0.0
    while t + 1e-6 < duration_sec:
        starts.append(t)
        t += hop_seconds
        if segment_seconds >= duration_sec:
            break
        if t >= duration_sec:
            break
    if not starts:
        starts = [0.0]
    return np.asarray(starts, dtype=np.float32)


def extract_segment_features(
    y: np.ndarray,
    sr: int,
    cfg: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or {}
    audio_cfg = cfg.get("audio", {})
    data_cfg = cfg.get("data", {})
    n_mels = int(data_cfg.get("n_mels", 128))
    n_chroma = int(data_cfg.get("n_chroma", 12))
    n_mfcc = int(audio_cfg.get("n_mfcc", 13))
    seg_s = float(audio_cfg.get("segment_seconds", 5.0))
    hop_s = float(audio_cfg.get("hop_seconds", 2.5))
    hop_length = 512

    y = normalize_track(y)
    mel = log_mel_spectrogram(y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    chroma = chroma_features(y, sr=sr, n_chroma=n_chroma, hop_length=hop_length)
    mfcc = mfcc_features(y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    duration = len(y) / float(sr)
    starts = segment_indices(duration, seg_s, hop_s)
    n_frames = mel.shape[1]
    times = frame_times(n_frames, sr, hop_length)

    feats = []
    for start in starts:
        end = start + seg_s
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            idx = int(np.argmin(np.abs(times - start)))
            mask = np.zeros(n_frames, dtype=bool)
            mask[idx] = True
        mel_seg = mel[:, mask]
        chr_seg = chroma[:, mask]
        mfcc_seg = mfcc[:, mask]
        vec = np.concatenate(
            [
                mel_seg.mean(axis=1),
                mel_seg.std(axis=1),
                chr_seg.mean(axis=1),
                chr_seg.std(axis=1),
                mfcc_seg.mean(axis=1),
                mfcc_seg.std(axis=1),
            ]
        ).astype(np.float32)
        feats.append(vec)

    node_features = np.stack(feats, axis=0)
    node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "node_features": node_features,
        "starts": starts,
        "mel": mel,
        "chroma": chroma,
        "mfcc": mfcc,
        "duration": np.float32(duration),
    }


def load_and_extract(path: Union[str, Path], cfg: dict) -> Dict[str, np.ndarray]:
    sr = int(cfg.get("data", {}).get("sample_rate", 22050))
    y, sr = load_audio(path, sr=sr)
    out = extract_segment_features(y, sr, cfg)
    out["waveform_len"] = np.int32(len(y))
    return out
