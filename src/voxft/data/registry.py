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


SOURCES: list[Source] = [
    # ---- 泰语 ----
    Source(
        "porjai_th", "th", "CMKL Porjai 标准泰语（700h，TTS 专用，主力语料）",
        "hf_dataset", "CMKL/Porjai-Thai-voice-dataset-central", "", "train",
        "CC-BY-SA-4.0", "录音棚级干净语料；体积大，建议先小样本试跑；无说话人列，加工时 ref_audio 比例用 0",
    ),
    Source(
        "fleurs_th", "th", "FLEURS 泰语（~12h，干净朗读）",
        "hf_dataset", "google/fleurs", "th_th", "train",
        "CC-BY-4.0", "高质锚点，也适合作验证集来源；无说话人列，加工时 ref_audio 比例用 0",
    ),
    Source(
        "thai20k", "th", "hotdogs/thai-speech-20k（1-10 万条）",
        "hf_dataset", "hotdogs/thai-speech-20k", "", "train",
        "CC-BY-4.0", "补充语料；仅 text/audio 列，无说话人，加工时 ref_audio 比例用 0",
    ),
    Source(
        "cv22_th", "th", "Common Voice 22 泰语（量大但噪）",
        "hf_dataset", "fsicoli/common_voice_22_0", "th", "train",
        "CC0", "官方已撤架，此为社区镜像（无需同意条款）；有 client_id 可做同说话人配对；务必走质检",
    ),
    # ---- Tagalog ----
    Source(
        "fleurs_tl", "tl", "FLEURS Tagalog（~12h，干净朗读）",
        "hf_dataset", "google/fleurs", "fil_ph", "train",
        "CC-BY-4.0", "高质锚点；无说话人列，加工时 ref_audio 比例用 0",
    ),
    Source(
        "cv22_tl", "tl", "Common Voice 22 Tagalog（量大）",
        "hf_dataset", "fsicoli/common_voice_22_0", "tl", "train",
        "CC0", "官方已撤架，此为社区镜像（无需同意条款）；主力体量来源；有 client_id；务必走质检",
    ),
    Source(
        "tagalog_tts", "tl", "welyjesch/tagalog_tts（1K-10K 条，许可待确认）",
        "hf_dataset", "welyjesch/tagalog_tts", "", "train",
        "未知", "仅 audio 列、无文本，暂不可直接用（需先 Whisper 转写）；商用前先核实许可",
    ),
    Source(
        "filipino_emotion", "tl", "filipino-emotion-tts（1-10 万条，情感语音）",
        "hf_dataset", "danielquillanroxas/filipino-emotion-tts", "", "train",
        "未知", "仅 audio+情绪标签、无文本，暂不可直接用（需先 Whisper 转写）；商用前先核实许可",
    ),
    # ---- 中文（混合防遗忘，建议占比 10-20%） ----
    Source(
        "aishell3", "zh", "AISHELL-3（~440h 多说话人朗读，录音棚级）",
        "openslr", "https://www.openslr.org/resources/93/data_aishell3.tgz", "", "",
        "Apache-2.0", "约 20GB，下载耗时；中文混合首选",
    ),
    Source(
        "fleurs_zh", "zh", "FLEURS 普通话（~10h，干净朗读）",
        "hf_dataset", "google/fleurs", "cmn_hans_cn", "train",
        "CC-BY-4.0", "少量高质补充；无说话人列，加工时 ref_audio 比例用 0",
    ),
]


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(f"未知数据源: {source_id}，可选: {[s.id for s in SOURCES]}")
