from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    id: str
    lang: str  # th / tl / zh
    label: str
    kind: str  # hf_dataset | openslr
    repo: str = ""
    config: str = ""
    split: str = "train"
    license: str = ""
    note: str = ""
    has_speaker: bool = False   # 有说话人列 → 加工自动按 0.4 配对，否则 0
    qc: str = "none"            # none / whisper / full(+UTMOS)；UTMOS 权重源已失效，默认仅 whisper
    short_clips: bool = False   # 单条普遍 <3s → 加工时自动同说话人拼接到 ~10s
    needs_transcribe: bool = False  # 无文本列 → 加工时先自动 Whisper 转写 + 语种过滤


SOURCES: list[Source] = [
    # ---- 泰语 ----
    Source(
        "porjai_th", "th", "CMKL Porjai 标准泰语（700h，TTS 专用，主力语料）",
        "hf_dataset", "CMKL/Porjai-Thai-voice-dataset-central", "", "train",
        "CC-BY-SA-4.0", "录音棚级干净语料；体积大，建议先小样本试跑",
        has_speaker=False, qc="none",
    ),
    Source(
        "fleurs_th", "th", "FLEURS 泰语（~12h，干净朗读）",
        "hf_dataset", "google/fleurs", "th_th", "train",
        "CC-BY-4.0", "高质锚点，也适合作验证集来源",
        has_speaker=False, qc="none",
    ),
    Source(
        "thai20k", "th", "hotdogs/thai-speech-20k（1-10 万条）",
        "hf_dataset", "hotdogs/thai-speech-20k", "", "train",
        "CC-BY-4.0", "补充语料；质量未知，自动 Whisper 校验",
        has_speaker=False, qc="whisper",
    ),
    Source(
        "cv22_th", "th", "Common Voice 22 泰语（量大但噪）",
        "hf_dataset", "fsicoli/common_voice_22_0", "th", "train",
        "CC0", "官方已撤架，此为社区镜像（无需同意条款）；众包噪音大，自动 Whisper 校验",
        has_speaker=True, qc="whisper",
    ),
    # ---- Tagalog ----
    Source(
        "fleurs_tl", "tl", "FLEURS Tagalog（~12h，干净朗读）",
        "hf_dataset", "google/fleurs", "fil_ph", "train",
        "CC-BY-4.0", "高质锚点",
        has_speaker=False, qc="none",
    ),
    Source(
        "filipino_speech", "tl", "filipinospeechcorpus（~27 万条，Tagalog 主力）",
        "hf_dataset", "sapinsapin/filipinospeechcorpus", "", "train",
        "MIT", "有 speaker_id；众包质量参差，自动 Whisper 校验；单条极短（中位 0.6s），自动拼接",
        has_speaker=True, qc="whisper", short_clips=True,
    ),
    Source(
        "tagalog_tts", "tl", "welyjesch/tagalog_tts（1K-10K 条，许可待确认）",
        "hf_dataset", "welyjesch/tagalog_tts", "", "train",
        "未知", "仅 audio 列，加工自动转写；商用前先核实许可",
        has_speaker=False, qc="none", needs_transcribe=True,
    ),
    Source(
        "filipino_emotion", "tl", "filipino-emotion-tts（1-10 万条，情感语音，短剧风格补充）",
        "hf_dataset", "danielquillanroxas/filipino-emotion-tts", "", "train",
        "未知", "情感语料（口语/情绪表达）；无文本列，加工自动转写；商用前先核实许可",
        has_speaker=False, qc="none", needs_transcribe=True,
    ),
    # ---- 中文（混合防遗忘，建议占比 10-20%） ----
    Source(
        "aishell3", "zh", "AISHELL-3（~440h 多说话人朗读，录音棚级）",
        "openslr", "https://www.openslr.org/resources/93/data_aishell3.tgz", "", "",
        "Apache-2.0", "约 20GB，下载耗时；中文混合首选",
        has_speaker=True, qc="none",
    ),
    Source(
        "fleurs_zh", "zh", "FLEURS 普通话（~10h，干净朗读）",
        "hf_dataset", "google/fleurs", "cmn_hans_cn", "train",
        "CC-BY-4.0", "少量高质补充",
        has_speaker=False, qc="none",
    ),
]


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(f"未知数据源: {source_id}，可选: {[s.id for s in SOURCES]}")
