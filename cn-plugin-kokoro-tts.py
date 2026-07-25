"""Kokoro TTS plugin for COVAS:NEXT using kokoro-onnx."""

from typing import Any, Iterable, Optional, override
import os
import re
import sys
import threading

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEPS_DIR = os.path.join(PLUGIN_DIR, "deps")

if os.path.isdir(DEPS_DIR) and DEPS_DIR not in sys.path:
    sys.path.insert(0, DEPS_DIR)

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from lib.PluginHelper import TTSModel
from lib.PluginSettingDefinitions import (
    ModelProviderDefinition,
    NumericalSetting,
    ParagraphSetting,
    PluginSettings,
    SelectOption,
    SelectSetting,
    SettingsGrid,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log

MODEL_FILENAME = "kokoro-v1.0.fp16.onnx"
VOICES_FILENAME = "voices-v1.0.bin"
TARGET_SAMPLE_RATE = 24000
STREAM_CHUNK_SAMPLES = 2400
DEFAULT_VOICE = "af_nova"
DEFAULT_ONNX_THREADS = max(1, (os.cpu_count() or 1) // 2)

VOICE_LABELS = {
    "af_heart": "American English - Heart",
    "af_alloy": "American English - Alloy",
    "af_aoede": "American English - Aoede",
    "af_bella": "American English - Bella",
    "af_jessica": "American English - Jessica",
    "af_kore": "American English - Kore",
    "af_nicole": "American English - Nicole",
    "af_nova": "American English - Nova",
    "af_river": "American English - River",
    "af_sarah": "American English - Sarah",
    "af_sky": "American English - Sky",
    "am_adam": "American English - Adam",
    "am_echo": "American English - Echo",
    "am_eric": "American English - Eric",
    "am_fenrir": "American English - Fenrir",
    "am_liam": "American English - Liam",
    "am_michael": "American English - Michael",
    "am_onyx": "American English - Onyx",
    "am_puck": "American English - Puck",
    "am_santa": "American English - Santa",
    "bf_alice": "British English - Alice",
    "bf_emma": "British English - Emma",
    "bf_isabella": "British English - Isabella",
    "bf_lily": "British English - Lily",
    "bm_daniel": "British English - Daniel",
    "bm_fable": "British English - Fable",
    "bm_george": "British English - George",
    "bm_lewis": "British English - Lewis",
    "ef_dora": "Spanish - Dora",
    "em_alex": "Spanish - Alex",
    "em_santa": "Spanish - Santa",
    "ff_siwis": "French - Siwis",
    "hf_alpha": "Hindi - Alpha",
    "hf_beta": "Hindi - Beta",
    "hm_omega": "Hindi - Omega",
    "hm_psi": "Hindi - Psi",
    "if_sara": "Italian - Sara",
    "im_nicola": "Italian - Nicola",
    "jf_alpha": "Japanese - Alpha",
    "jf_gongitsune": "Japanese - Gongitsune",
    "jf_nezumi": "Japanese - Nezumi",
    "jf_tebukuro": "Japanese - Tebukuro",
    "jm_kumo": "Japanese - Kumo",
    "pf_dora": "Brazilian Portuguese - Dora",
    "pm_alex": "Brazilian Portuguese - Alex",
    "pm_santa": "Brazilian Portuguese - Santa",
    "zf_xiaobei": "Mandarin Chinese - Xiaobei",
    "zf_xiaoni": "Mandarin Chinese - Xiaoni",
    "zf_xiaoxiao": "Mandarin Chinese - Xiaoxiao",
    "zf_xiaoyi": "Mandarin Chinese - Xiaoyi",
    "zm_yunjian": "Mandarin Chinese - Yunjian",
    "zm_yunxi": "Mandarin Chinese - Yunxi",
    "zm_yunxia": "Mandarin Chinese - Yunxia",
    "zm_yunyang": "Mandarin Chinese - Yunyang",
}
VOICE_LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}
VOICE_ALIASES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "shimmer": "af_bella",
}


def _text_chunks(text: str, max_characters: int = 400) -> list[str]:
    """Keep each inference request within Kokoro's reliable input range."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_characters:
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > max_characters:
                    chunks.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            continue
        if current and len(current) + len(sentence) + 1 > max_characters:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


class KokoroTTSModel(TTSModel):
    """Kokoro Text-to-Speech model implementation."""

    def __init__(self, model_dir: str, default_voice: str, speed: float, onnx_threads: int):
        super().__init__("kokoro-tts")
        self.model_dir = model_dir
        self.default_voice = default_voice
        self.speed = min(max(speed, 0.5), 2.0)
        self.onnx_threads = max(1, int(onnx_threads))
        self._tts: Optional[Kokoro] = None
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()

    def _load_model(self) -> Kokoro:
        if self._tts is not None:
            return self._tts

        with self._load_lock:
            if self._tts is not None:
                return self._tts

            model_path = os.path.join(self.model_dir, MODEL_FILENAME)
            voices_path = os.path.join(self.model_dir, VOICES_FILENAME)
            missing = [path for path in (model_path, voices_path) if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    "Missing Kokoro model assets. Run the included downloader or reinstall the plugin. "
                    f"Missing: {', '.join(missing)}"
                )

            log("info", f"Loading Kokoro model from {model_path}")
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.onnx_threads
            session_options.inter_op_num_threads = 1
            providers = [os.environ.get("ONNX_PROVIDER", "CPUExecutionProvider")]
            session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            self._tts = Kokoro.from_session(session, voices_path)
            return self._tts

    def _resolve_voice(self, requested_voice: str) -> str:
        requested_voice = (requested_voice or "").strip().lower()
        if requested_voice in VOICE_LABELS:
            return requested_voice
        if requested_voice in VOICE_ALIASES:
            return VOICE_ALIASES[requested_voice]
        if requested_voice:
            log("warning", f"Unknown Kokoro voice '{requested_voice}', using '{self.default_voice}'")
        return self.default_voice

    @override
    def synthesize(self, text: str, voice: str) -> Iterable[bytes]:
        if not text.strip():
            return

        tts = self._load_model()
        target_voice = self._resolve_voice(voice)
        language = VOICE_LANGUAGES.get(target_voice[0], "en-us")
        text_passes = _text_chunks(text)

        with self._synthesis_lock:
            for text_pass in text_passes:
                samples, sample_rate = tts.create(
                    text_pass,
                    voice=target_voice,
                    speed=self.speed,
                    lang=language,
                )
                if sample_rate != TARGET_SAMPLE_RATE:
                    raise RuntimeError(
                        f"Kokoro returned {sample_rate} Hz audio, expected {TARGET_SAMPLE_RATE} Hz"
                    )

                pcm = (np.asarray(samples, dtype=np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
                for start in range(0, len(pcm), STREAM_CHUNK_SAMPLES):
                    yield pcm[start : start + STREAM_CHUNK_SAMPLES].tobytes()


class KokoroTTSPlugin(PluginBase):
    """Plugin providing Kokoro TTS services."""

    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        self.model_dir = os.path.join(PLUGIN_DIR, "model")
        self.settings_config = PluginSettings(
            key="Kokoro TTS",
            label="Kokoro TTS",
            icon="record_voice_over",
            grids=[
                SettingsGrid(
                    key="general",
                    label="General",
                    fields=[
                        ParagraphSetting(
                            key="info_text",
                            label=None,
                            type="paragraph",
                            readonly=False,
                            placeholder=None,
                            content=(
                                'To use Kokoro TTS, select it as your "TTS provider" in "Advanced" '
                                "-> \"TTS Settings\". Language is selected automatically from the voice preset."
                            ),
                        ),
                    ],
                ),
            ],
        )
        self.model_providers = [
            ModelProviderDefinition(
                kind="tts",
                id="kokoro-tts",
                label="Kokoro (Local)",
                settings_config=[
                    SettingsGrid(
                        key="settings",
                        label="Settings",
                        fields=[
                            SelectSetting(
                                key="voice",
                                label="Default voice",
                                type="select",
                                readonly=False,
                                placeholder=None,
                                default_value=DEFAULT_VOICE,
                                select_options=[
                                    SelectOption(key=name, label=label, value=name, disabled=False)
                                    for name, label in VOICE_LABELS.items()
                                ],
                                multi_select=False,
                            ),
                            NumericalSetting(
                                key="onnx_threads",
                                label="CPU Threads",
                                type="number",
                                readonly=False,
                                placeholder=str(DEFAULT_ONNX_THREADS),
                                default_value=DEFAULT_ONNX_THREADS,
                                min_value=1,
                                max_value=max(1, os.cpu_count() or 1),
                                step=1,
                            ),
                            NumericalSetting(
                                key="speed",
                                label="Speed",
                                type="number",
                                readonly=False,
                                placeholder="1.0",
                                default_value=1.0,
                                min_value=0.5,
                                max_value=2.0,
                                step=0.1,
                            ),
                        ],
                    ),
                ],
            )
        ]

    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> TTSModel:
        if provider_id != "kokoro-tts":
            raise ValueError(f"Unknown Kokoro provider: {provider_id}")

        default_voice = str(settings.get("voice", DEFAULT_VOICE))
        if default_voice not in VOICE_LABELS:
            default_voice = DEFAULT_VOICE
        speed = float(settings.get("speed", 1.0))
        onnx_threads = int(settings.get("onnx_threads", DEFAULT_ONNX_THREADS))
        return KokoroTTSModel(self.model_dir, default_voice, speed, onnx_threads)


if __name__ == "__main__":
    plugin_manifest = PluginManifest(
        name="Kokoro TTS Plugin",
        version="0.0.2",
        author="COVAS:NEXT",
        description="Kokoro TTS Plugin for COVAS:NEXT",
    )
    plugin = KokoroTTSPlugin(plugin_manifest)
    try:
        plugin.create_model("kokoro-tts", {})
        log("info", "Kokoro TTS Plugin initialized successfully.")
    except Exception as exc:
        log("error", f"Failed to initialize Kokoro TTS Plugin: {exc}")
