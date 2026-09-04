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
    has_speaker: bool = False   # 有说话人列 → 加工自动按 0.4 配对，否则 0（除非 pseudo_speaker）
    qc: str = "none"            # none / whisper / full(+UTMOS)；UTMOS 权重源已失效，默认仅 whisper
    concat_target: float = 0.0  # >0：短句语料同会话拼接到约该秒数（0=不拼）
    needs_transcribe: bool = False  # 有条目缺文本 → 加工时自动 Whisper 转写 + 语种过滤

    # ---- 列映射（留空则按通用规则探测）----
    audio_cols: tuple[str, ...] = ()    # 候选音频列，按顺序取第一个存在的
    text_cols: tuple[str, ...] = ()
    speaker_cols: tuple[str, ...] = ()
    emotion_col: str = ""               # 情绪标签列 → 写入 manifest 的 emotion 字段
    session_col: str = ""               # 同一次录音的分组列 → 拼接只在组内进行
    label_names: tuple[str, ...] = ()   # emotion_col 是 ClassLabel 整数时的取值表

    # ---- 行级过滤：(列, 操作, 值)，操作 in / not_in / >= / <= ----
    row_filters: tuple[tuple[str, str, object], ...] = ()

    # ---- 加工策略 ----
    expressive: bool = False       # 情感/口语语料：去念稿感主力，参与控制前缀生成
    pseudo_speaker: bool = False   # 无说话人列但值得聚类出伪说话人，以启用 ref 配对
    sent_sep: str = ""             # 拼接时的句间分隔（留空按语种取默认）

    def audio_column(self, columns) -> str | None:
        for c in self.audio_cols or ("audio",):
            if c in columns:
                return c
        return None

    def separator(self) -> str:
        if self.sent_sep:
            return self.sent_sep
        # 泰语句间用空格（其书写习惯无句点），中文用全角句号，其余用英文句点
        return {"th": " ", "zh": "。"}.get(self.lang, ". ")


SOURCES: list[Source] = [
    # ---- 泰语 ----
    Source(
        "thai_ser", "th", "THAI-SER 泰语情感语音（2.8 万条/41h，200 名演员，表现力主力）",
        "hf_dataset", "airesearch/thai-ser", "", "train",
        "CC-BY-SA-4.0",
        "有 actor_id（可 ref 配对）与情绪标签；impro 为即兴对话，去念稿感最值钱。"
        "整包 12.7GB（含 4 路麦克风），试跑请用 max_samples",
        has_speaker=True, qc="none", needs_transcribe=True,
        audio_cols=("mic_con", "mic_clip", "mic_middle"),  # 不用 mic_zoom（网络录音，质量差）
        text_cols=("script_sent",),      # impro 轮次无文本，加工时自动转写
        speaker_cols=("actor_id",),
        emotion_col="majority_emo",
        session_col="session_id",
        row_filters=(("agreement", ">=", 0.7),),  # 标注一致性低的丢掉
        expressive=True,
    ),
    Source(
        "porjai_th", "th", "CMKL Porjai 标准泰语（700h，TTS 专用，发音锚点）",
        "hf_dataset", "CMKL/Porjai-Thai-voice-dataset-central", "", "train",
        "CC-BY-SA-4.0", "录音棚级干净语料；体积大，建议先小样本试跑",
        has_speaker=False, qc="none",
    ),
    Source(
        "fleurs_th", "th", "FLEURS 泰语（~12h，干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "th_th", "train",
        "CC-BY-4.0", "高质锚点；无说话人列，加工时聚类伪说话人以启用 ref 配对",
        has_speaker=False, qc="none", pseudo_speaker=True,
    ),
    Source(
        "thai20k", "th", "hotdogs/thai-speech-20k（1-10 万条）",
        "hf_dataset", "hotdogs/thai-speech-20k", "", "train",
        "CC-BY-4.0", "补充语料；质量未知，自动 Whisper 校验",
        has_speaker=False, qc="whisper", pseudo_speaker=True,
    ),
    Source(
        "cv22_th", "th", "Common Voice 22 泰语（量大但噪，发音锚点）",
        "hf_dataset", "fsicoli/common_voice_22_0", "th", "train",
        "CC0", "官方已撤架，此为社区镜像（无需同意条款）；众包噪音大，自动 Whisper 校验",
        has_speaker=True, qc="whisper",
    ),
    # ---- Tagalog ----
    Source(
        "filipino_emotion", "tl", "filipino-emotion-tts（1.1 万条，6 情绪，表现力主力）",
        "hf_dataset", "danielquillanroxas/filipino-emotion-tts", "", "train",
        "未知", "中位 3.0s，带情绪标签；无文本列，加工自动转写；商用前先核实许可",
        has_speaker=False, qc="none", needs_transcribe=True,
        emotion_col="label",
        label_names=("angry", "fearful", "happy", "neutral", "sad", "surprised"),
        expressive=True, pseudo_speaker=True,
    ),
    Source(
        "fleurs_tl", "tl", "FLEURS Tagalog（~12h，干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "fil_ph", "train",
        "CC-BY-4.0", "高质锚点；无说话人列，加工时聚类伪说话人",
        has_speaker=False, qc="none", pseudo_speaker=True,
    ),
    Source(
        "filipino_speech", "tl", "filipinospeechcorpus（22 万条，绝大部分是孤立单词）",
        "hf_dataset", "sapinsapin/filipinospeechcorpus", "", "train",
        "MIT",
        "⚠️ 中位 0.63s / num_words 中位 1：直接拼接会训出报菜名式念稿感。"
        "已自动过滤掉 speech_type=machine 与 num_words<4，剩余部分按 source_file 同会话拼到 ~6s",
        has_speaker=True, qc="whisper", concat_target=6.0,
        speaker_cols=("speaker_id",),
        session_col="source_file",
        row_filters=(("speech_type", "not_in", ("machine",)),
                     ("num_words", ">=", 4)),
    ),
    Source(
        "tagalog_tts", "tl", "welyjesch/tagalog_tts（1K-10K 条，许可待确认）",
        "hf_dataset", "welyjesch/tagalog_tts", "", "train",
        "未知", "仅 audio 列，加工自动转写；商用前先核实许可",
        has_speaker=False, qc="none", needs_transcribe=True, pseudo_speaker=True,
    ),
    # ---- 中文（混合防遗忘，建议占比 10-20%） ----
    Source(
        "aishell3", "zh", "AISHELL-3（~440h 多说话人朗读，录音棚级）",
        "openslr", "https://www.openslr.org/resources/93/data_aishell3.tgz", "", "",
        "Apache-2.0", "约 20GB，下载耗时；中文混合首选（有说话人，可 ref 配对）",
        has_speaker=True, qc="none",
    ),
    Source(
        "fleurs_zh", "zh", "FLEURS 普通话（~10h，干净朗读）",
        "hf_dataset", "google/fleurs", "cmn_hans_cn", "train",
        "CC-BY-4.0", "少量高质补充",
        has_speaker=False, qc="none", pseudo_speaker=True,
    ),
]


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(f"未知数据源: {source_id}，可选: {[s.id for s in SOURCES]}")


def row_passes(src: Source, get) -> bool:
    """行级过滤；get(col) 返回该列的值（缺列返回 None，视为通过）。"""
    for col, op, val in src.row_filters:
        v = get(col)
        if v is None:
            continue
        if op == "in" and v not in val:
            return False
        if op == "not_in" and v in val:
            return False
        if op in (">=", "<="):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if op == ">=" and fv < float(val):
                return False
            if op == "<=" and fv > float(val):
                return False
    return True
