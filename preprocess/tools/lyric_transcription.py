# https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
import os
import re
import sys
import time
import types
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
from funasr import AutoModel


def _ensure_nv_one_logger_stub():
    """Register stub modules for nv_one_logger if not installed.

    NeMo 2.6.x unconditionally imports ``nv_one_logger`` (an NVIDIA-internal
    package) during ``import nemo.collections.asr``.  When this package is
    absent the import chain fails with ``ModuleNotFoundError``.

    This helper injects lightweight stub modules into ``sys.modules`` so the
    import succeeds without the real package.  The stubs provide only the
    names that NeMo actually references at import time.
    """
    if "nv_one_logger" in sys.modules:
        return

    # Module paths that NeMo's one_logger_callback.py imports:
    #   nv_one_logger.api.config              -> OneLoggerConfig
    #   nv_one_logger.training_telemetry.api.callbacks  -> on_app_start
    #   nv_one_logger.training_telemetry.api.config     -> TrainingTelemetryConfig
    #   nv_one_logger.training_telemetry.api.training_telemetry_provider -> TrainingTelemetryProvider
    #   nv_one_logger.training_telemetry.integration.pytorch_lightning   -> TimeEventCallback
    sub_paths = [
        "nv_one_logger",
        "nv_one_logger.api",
        "nv_one_logger.api.config",
        "nv_one_logger.training_telemetry",
        "nv_one_logger.training_telemetry.api",
        "nv_one_logger.training_telemetry.api.callbacks",
        "nv_one_logger.training_telemetry.api.config",
        "nv_one_logger.training_telemetry.api.training_telemetry_provider",
        "nv_one_logger.training_telemetry.integration",
        "nv_one_logger.training_telemetry.integration.pytorch_lightning",
    ]
    for path in sub_paths:
        if path not in sys.modules:
            sys.modules[path] = types.ModuleType(path)

    # Provide dummy classes / functions referenced at import time.
    from lightning.pytorch.callbacks import Callback as _PTLCallback

    class _DummyOneLoggerConfig:
        def __init__(self, **kw):
            pass

    class _DummyTrainingTelemetryConfig:
        def __init__(self, **kw):
            pass

    class _DummyProvider:
        _inst = None

        @classmethod
        def instance(cls):
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

        def with_base_config(self, *a, **kw):
            return self

        def with_export_config(self, *a, **kw):
            return self

        def configure_provider(self, *a, **kw):
            return self

        def set_training_telemetry_config(self, *a, **kw):
            pass

        class config:
            telemetry_config = None

    class _DummyTimeEventCallback(_PTLCallback):
        def __init__(self, *a, **kw):
            super().__init__()

    sys.modules["nv_one_logger.api.config"].OneLoggerConfig = _DummyOneLoggerConfig
    sys.modules["nv_one_logger.training_telemetry.api.callbacks"].on_app_start = lambda *a, **kw: None
    sys.modules["nv_one_logger.training_telemetry.api.config"].TrainingTelemetryConfig = _DummyTrainingTelemetryConfig
    sys.modules["nv_one_logger.training_telemetry.api.training_telemetry_provider"].TrainingTelemetryProvider = _DummyProvider
    sys.modules["nv_one_logger.training_telemetry.integration.pytorch_lightning"].TimeEventCallback = _DummyTimeEventCallback


def _build_words_with_gaps(raw_words, raw_timestamps, wav_fn: str):
    words, word_durs = [], []
    prev = 0.0
    for w, t in zip(raw_words, raw_timestamps):
        s, e = float(t[0]), float(t[1])
        if s > prev:
            words.append("<SP>")
            word_durs.append(s - prev)
        words.append(w)
        word_durs.append(e - s)
        prev = e

    wav_len = librosa.get_duration(filename=wav_fn)
    if wav_len > prev:
        if len(words) == 0:
            words.append("<SP>")
            word_durs.append(wav_len)
            return words, word_durs
        if words[-1] != "<SP>":
            words.append("<SP>")
            word_durs.append(wav_len - prev)
        else:
            word_durs[-1] += wav_len - prev

    return words, word_durs

def _word_dur_post_process(words, word_durs, f0):
    """Post-process word durations using f0 to better place silences.
    """
    # f0 time grid parameters
    sr = 24000  # f0 sample rate
    hop_length = 480  # f0 hop length

    # Convert word durations (seconds) to frame boundaries on the f0 grid.
    boundaries = np.cumsum([
        0,
        *[
            int(dur * sr / hop_length)
            for dur in word_durs
        ],
    ]).tolist()

    sil_tolerance = 5   # tolerance frames for silence detection
    ext_tolerance = 5   # tolerance frames for vocal extension

    new_words: list[str] = []
    new_word_durs: list[float] = []
    if words:
        new_words.append(words[0])
        new_word_durs.append(word_durs[0])

    for i in range(1, len(words)):
        word = words[i]
        if word == "<SP>":
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]

            num_frames = end_frame - start_frame
            frame_idx = start_frame

            # Find first region with at least 5 consecutive "unvoiced" frames.
            unvoiced_count = 0
            while frame_idx < end_frame:
                if f0[frame_idx] <= 1:  # unvoiced
                    unvoiced_count += 1
                    if unvoiced_count >= sil_tolerance:
                        frame_idx -= sil_tolerance - 1  # back to the last voiced frame
                        break
                else:
                    unvoiced_count = 0
                frame_idx += 1

            voice_frames = frame_idx - start_frame

            if voice_frames >= int(num_frames * 0.9):  # over 90% voiced
                # Treat the whole "<SP>" as silence and merge into previous word.
                new_word_durs[-1] += word_durs[i]
            elif voice_frames >= ext_tolerance:  # over 5 frames voiced
                # Split the "<SP>" into two parts: leading silence and tail kept as "<SP>".
                dur = voice_frames * hop_length / sr
                new_word_durs[-1] += dur
                new_words.append("<SP>")
                new_word_durs.append(word_durs[i] - dur)
            else:
                # Too short to adjust, keep as-is.
                new_words.append(word)
                new_word_durs.append(word_durs[i])
        else:
            new_words.append(word)
            new_word_durs.append(word_durs[i])

    return new_words, new_word_durs


class _ASRZhModel:
    """Mandarin/Cantonese ASR wrapper."""

    def __init__(self, model_path: str, device: str):
        self.model = AutoModel(
            model=model_path,
            disable_update=True,
            device=device,
        )

    def process(self, wav_fn):
        out = self.model.generate(wav_fn, output_timestamp=True)[0]
        raw_words = out["text"].replace("@", "").split(" ")
        raw_timestamps = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]
        words, word_durs = _build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        if os.path.exists(wav_fn.replace(".wav", "_f0.npy")):
            words, word_durs = _word_dur_post_process(
                words, word_durs, np.load(wav_fn.replace(".wav", "_f0.npy"))
            )

        return words, word_durs


class _ASREnModel:
    """English ASR wrapper for NeMo Parakeet-TDT."""

    def __init__(self, model_path: str, device: str):
        try:
            _ensure_nv_one_logger_stub()
            import nemo.collections.asr as nemo_asr  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "NeMo (nemo_toolkit) is required for ASR English but is not available in this Python env. "
                "Install it in the active environment, then retry."
            ) from e

        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=device,
        )
        # Disable CUDA graphs for decoding — avoids cuda-bindings API
        # incompatibility between NeMo 2.6.x and newer PyTorch/CUDA versions.
        # Setting this in cfg.decoding.greedy ensures it persists across
        # change_decoding_strategy() calls (e.g. when timestamps=True).
        from omegaconf import open_dict
        with open_dict(self.model.cfg.decoding):
            if not self.model.cfg.decoding.get("greedy"):
                self.model.cfg.decoding.greedy = {}
            self.model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        self.model.change_decoding_strategy(self.model.cfg.decoding, verbose=False)
        self.model.eval()

    @staticmethod
    def _clean_word(word: str) -> str:
        return re.sub(r"[\?\.,:]", "", word).strip()

    @staticmethod
    def _extract_word_segments(output: Any) -> List[Dict[str, Any]]:
        ts = getattr(output, "timestamp", None)
        if not ts or not isinstance(ts, dict):
            return []
        word_ts = ts.get("word")
        return word_ts if isinstance(word_ts, list) else []

    def process(self, wav_fn: str) -> Tuple[List[str], List[float]]:
        outputs = self.model.transcribe(
            [wav_fn],
            timestamps=True,
            batch_size=1,
            num_workers=0,
        )
        output = outputs[0] if outputs else None

        raw_words: List[str] = []
        raw_timestamps: List[List[float]] = []
        if output is not None:
            for w in self._extract_word_segments(output):
                s, e = float(w.get("start", 0.0)), float(w.get("end", 0.0))
                word = self._clean_word(str(w.get("word", "")))
                if word:
                    raw_words.append(word)
                    raw_timestamps.append([s, e])

        words, durs = _build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        if os.path.exists(wav_fn.replace(".wav", "_f0.npy")):
            words, durs = _word_dur_post_process(
                words, durs, np.load(wav_fn.replace(".wav", "_f0.npy"))
            )

        return words, durs


class _ASRJaModel:
    """Japanese ASR wrapper using faster-whisper."""

    def __init__(self, model_size: str, device: str):
        from faster_whisper import WhisperModel

        compute_type = "float16" if "cuda" in device else "int8"
        self.model = WhisperModel(model_size, device=device.split(":")[0], compute_type=compute_type)

    @staticmethod
    def _clean_word(word: str) -> str:
        return re.sub(r"[\?\.,:]", "", word).strip()

    def process(self, wav_fn: str) -> Tuple[List[str], List[float]]:
        segments, _ = self.model.transcribe(wav_fn, language="ja", word_timestamps=True)

        raw_words: List[str] = []
        raw_timestamps: List[List[float]] = []
        for segment in segments:
            if segment.words is None:
                continue
            for w in segment.words:
                word = self._clean_word(w.word)
                if word:
                    raw_words.append(word)
                    raw_timestamps.append([w.start, w.end])

        words, durs = _build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        if os.path.exists(wav_fn.replace(".wav", "_f0.npy")):
            words, durs = _word_dur_post_process(
                words, durs, np.load(wav_fn.replace(".wav", "_f0.npy"))
            )

        return words, durs


class LyricTranscriber:
    """Transcribe lyrics from singing voice segment
    """

    def __init__(
        self,
        zh_model_path: str,
        en_model_path: str,
        device: str = "cuda",
        *,
        ja_model_size: str = "large-v3",
        verbose: bool = True,
    ):
        """Initialize lyric transcriber.

        Args:
            zh_model_path (str): Path to the Chinese model file.
            en_model_path (str): Path to the English model file.
            device (str): Device to use for tensor operations.
            ja_model_size (str): Whisper model size for Japanese ASR.
            verbose (bool): Whether to print verbose logs.
        """
        self.verbose = verbose
        self.device = device
        self.zh_model_path = zh_model_path
        self.en_model_path = en_model_path
        self.ja_model_size = ja_model_size

        if self.verbose:
            print(
                "[lyric transcription] init: start:",
                f"device={device}",
                f"model_path={zh_model_path}",
            )

        # Always initialize Chinese ASR.
        self.zh_model = _ASRZhModel(device=device, model_path=zh_model_path)

        # English and Japanese ASR will be lazily initialized on first request
        self.en_model = None
        self.ja_model = None

        if self.verbose:
            print("[lyric transcription] init: success")

    def process(self, wav_fn, language: str | None = "Mandarin", *, verbose: bool | None = None):
        """ Lyric transcriber process

        Args:
            wav_fn (str): Path to the audio file.
            language (str | None): Language of the audio. Defaults to "Mandarin". Supports "Mandarin", "Cantonese" and "English".
            verbose (bool | None): Whether to print verbose logs. Defaults to None.
        """
        v = self.verbose if verbose is None else verbose
        if language not in {"Mandarin", "Cantonese", "English", "Japanese"}:
            raise ValueError(f"Unsupported language: {language}, should be one of ['Mandarin', 'Cantonese', 'English', 'Japanese']")
        if v:
            print(f"[lyric transcription] process: start: wav_fn={wav_fn} language={language}")
            t0 = time.time()

        lang = (language or "auto").lower()
        if lang in {"japanese"}:
            if self.ja_model is None:
                if v:
                    print("[lyric transcription] init Japanese ASR (faster-whisper)")
                self.ja_model = _ASRJaModel(model_size=self.ja_model_size, device=self.device)
            out = self.ja_model.process(wav_fn)
        elif lang in {"english"}:
            if self.en_model is None:
                # Lazy-load NeMo model only when English is actually used.
                if v:
                    print("[lyric transcription] init English ASR, please make sure NeMo is installed")
                self.en_model = _ASREnModel(model_path=self.en_model_path, device=self.device)
            out = self.en_model.process(wav_fn)
        else:
            out = self.zh_model.process(wav_fn)

        if v:
            words, durs = out
            n_words = len(words) if isinstance(words, list) else 0
            dur_sum = float(sum(durs)) if isinstance(durs, list) else 0.0
            dt = time.time() - t0
            print(
                "[lyric transcription] process: done:",
                f"n_words={n_words}",
                f"dur_sum={dur_sum:.3f}s",
                f"time={dt:.3f}s",
            )

        return out


if __name__ == "__main__":
    m = LyricTranscriber(
        zh_model_path="pretrained_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        en_model_path="pretrained_models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        device="cuda"
    )
    print(m.process("example/test/asr_zh.wav", language="Mandarin"))
    print(m.process("example/test/asr_en.wav", language="English"))