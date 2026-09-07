from __future__ import annotations

import math
from dataclasses import dataclass


PREFERRED_TAG = "【首选】"


@dataclass(frozen=True)
class Source:
    id: str
    lang: str  # th / tl / zh / en
    label: str
    kind: str  # hf_dataset | openslr | local
    repo: str = ""
    config: str = ""
    split: str = "train"
    license: str = ""
    note: str = ""
    has_speaker: bool = False   # 有可靠的说话人身份；视频 ID 不算
    qc: str = "none"            # none / whisper / full(+UTMOS)；UTMOS 权重源已失效，默认仅 whisper
    needs_transcribe: bool = False  # 有条目缺文本 → 加工时自动 Whisper 转写 + 语种过滤

    # ---- 列映射（留空则按通用规则探测）----
    audio_cols: tuple[str, ...] = ()    # 候选音频列，按顺序取第一个存在的
    text_cols: tuple[str, ...] = ()
    speaker_cols: tuple[str, ...] = ()
    emotion_col: str = ""               # 情绪标签列 → 写入 manifest 的 emotion 字段
    session_col: str = ""               # 同一次录音的分组列，用于训练/验证隔离
    session_prefix_sep: str = ""        # YODAS 从右侧移除序号与两个时间戳，保留完整视频 ID
    label_names: tuple[str, ...] = ()   # emotion_col 是 ClassLabel 整数时的取值表

    # ---- 行级过滤：(列, 操作, 值)，操作 in / not_in / >= / <= ----
    row_filters: tuple[tuple[str, str, object], ...] = ()

    # ---- 加工策略 ----
    accept_langs: tuple[str, ...] = ()  # 允许的检测语种（留空=按 lang 推导）
    role: str = "anchor"           # expressive 表现力主力 / anchor 发音·口语锚点 / antiforget 中文防遗忘
    preferred: bool = False        # 该语种该角色下的首选源
    expressive: bool = False       # 情感/口语语料：去念稿感主力，参与控制前缀生成
    pseudo_speaker: bool = False   # 仅辅助审计，不作为 ref 身份依据
    quality: int = 0               # 同语种下 UI 排序，越高越优先

    def audio_column(self, columns) -> str | None:
        for c in self.audio_cols or ("audio",):
            if c in columns:
                return c
        return None

    def languages(self) -> tuple[str, ...]:
        """Whisper 转写/质检时允许的语种。

        Tagalog 默认放行 en：短剧台词是 Taglish，句内英文词多的样本 Whisper 常判成
        en，只认 tl 会把最该保留的 code-switch 样本全部误杀。
        """
        if self.accept_langs:
            return self.accept_langs
        return ("tl", "en") if self.lang == "tl" else (self.lang,)

    def session_of(self, value) -> str:
        v = "" if value is None else str(value).strip()
        if v.lower() in ("none", "nan", "null"):
            return ""
        if v and self.session_prefix_sep:
            v = v.rsplit(self.session_prefix_sep, 3)[0]
        return v

    def display(self) -> str:
        """下拉框显示文本；反解回 id 一律走 source_id_from_display，别自己 split。"""
        tag = f" {PREFERRED_TAG}" if self.preferred else ""
        return f"{self.id}{tag} — {self.label} [{self.license}]"

SOURCES: list[Source] = [
    Source("drama_tl", "tl", "已审核真人 Tagalog/Taglish 短剧对白（自备 JSONL）", "local",
           license="按自有授权", note="填写 speaker_verified=true；英文/中文同人参考可标 reference_only=true",
           has_speaker=True, role="expressive", preferred=True, expressive=True, quality=100),
    Source("drama_th", "th", "已审核真人泰语短剧对白（自备 JSONL）", "local",
           license="按自有授权", has_speaker=True, role="expressive", expressive=True, quality=100),
    Source("replay_en", "en", "英文多说话人回放（自备已审核 JSONL，如 VCTK）", "local",
           license="按原数据授权", has_speaker=True, role="antiforget", quality=100),
    # ---- 泰语 ----
    Source(
        "thai_ser", "th", "THAI-SER 泰语情感语音（2.8 万条/41h，200 名演员，5 情绪）",
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
        row_filters=(("agreement", ">=", 0.7), ("turn_type", "in", ("impro",))),
        role="expressive", preferred=True, expressive=True, quality=90,
    ),
    Source(
        "yodas_th", "th", "YODAS2-Sidon 泰语 TTS 精选（14 万条/156h，YouTube 视频语音）",
        "hf_dataset", "Chalermdej/yodas2_sidon_th_tts", "", "train",
        "CC-BY-3.0",
        "口语候选：来自 YouTube 视频，需筛选自然对白、排除多人重叠；已有 DNSMOS + 三路 ASR 分级。"
        "speaker_id 是视频级近似身份，默认不配 ref；上游逐条峰值归一，不生成音量指令。"
        "保留完整 3-30s 片段，不自动拼接；整包 26.6GB，试跑用 max_samples",
        has_speaker=False, qc="none",
        text_cols=("text",), speaker_cols=("speaker_id",),
        session_col="utt_id", session_prefix_sep="-",
        row_filters=(("grade_avg", "in", ("S+", "S")),
                     ("dnsmos_overall", ">=", 3.2)),
        role="anchor", preferred=True, quality=80,
    ),
    Source(
        "porjai_th", "th", "CMKL Porjai 标准泰语（700h，TTS 专用，录音棚朗读）",
        "hf_dataset", "CMKL/Porjai-Thai-voice-dataset-central", "", "train",
        "CC-BY-SA-4.0", "录音棚级干净语料；体积大，建议先小样本试跑",
        has_speaker=False, qc="none", quality=70,
    ),
    Source(
        "fleurs_th", "th", "FLEURS 泰语（~12h，干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "th_th", "train",
        "CC-BY-4.0", "朗读发音补充；无可靠说话人身份，不配 ref",
        has_speaker=False, qc="none", quality=60,
    ),
    Source(
        "thai20k", "th", "hotdogs/thai-speech-20k（1-10 万条）",
        "hf_dataset", "hotdogs/thai-speech-20k", "", "train",
        "CC-BY-4.0", "补充语料；质量未知，自动 Whisper 校验",
        has_speaker=False, qc="whisper", quality=40,
    ),
    Source(
        "cv22_th", "th", "Common Voice 22 泰语（量大但噪，发音锚点）",
        "hf_dataset", "fsicoli/common_voice_22_0", "th", "train",
        "CC0", "官方已撤架，此为社区镜像（无需同意条款）；众包噪音大，自动 Whisper 校验",
        has_speaker=True, qc="whisper", quality=30,
    ),
    # ---- Tagalog ----
    Source(
        "filipino_emotion", "tl", "filipino-emotion-tts（1.1 万条，6 情绪）",
        "hf_dataset", "danielquillanroxas/filipino-emotion-tts", "", "train",
        "未知", "待审计候选：缺文本/说话人/完整来源说明，核验真人录音和情绪标签后再使用",
        has_speaker=False, qc="none", needs_transcribe=True,
        emotion_col="label",
        label_names=("angry", "fearful", "happy", "neutral", "sad", "surprised"),
        role="expressive", expressive=True, quality=40,
    ),
    Source(
        "fleurs_tl", "tl", "FLEURS Tagalog（~12h，干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "fil_ph", "train",
        "CC-BY-4.0", "朗读发音补充；无可靠说话人身份，不配 ref",
        has_speaker=False, qc="none", quality=80,
    ),
    Source(
        "filipino_speech", "tl", "filipinospeechcorpus（22 万条，绝大部分是孤立单词）",
        "hf_dataset", "sapinsapin/filipinospeechcorpus", "", "train",
        "MIT",
        "⚠️ 中位 0.63s / num_words 中位 1：直接拼接会训出报菜名式念稿感。"
        "已过滤 speech_type=machine 与 num_words<4，仅保留完整句，不拼接孤立词",
        has_speaker=True, qc="whisper",
        speaker_cols=("speaker_id",),
        session_col="source_file", quality=60,
        row_filters=(("speech_type", "not_in", ("machine",)),
                     ("num_words", ">=", 4)),
    ),
    Source(
        "filswitch", "tl", "FilSwitch（2.7K 条 Taglish 语料，句内中英混杂）",
        "hf_dataset", "qwerttyuiiop/FilSwitch", "", "train",
        "未声明", "新闻朗读语料，仅作低占比 Taglish 发音补充；"
        "教的是句内英文词与数字怎么念（对应线上「RAW 被念成英文」类反馈），不是情绪。"
        "不做语种过滤，否则英文占比高的样本会被 Whisper 判成 en 而误杀；商用前先核实许可",
        has_speaker=False, qc="none",
        role="anchor", preferred=True, quality=50,
    ),
    Source(
        "tagalog_tts", "tl", "welyjesch/tagalog_tts（1K-10K 条，许可待确认）",
        "hf_dataset", "welyjesch/tagalog_tts", "", "train",
        "未知", "仅 audio 列，加工自动转写；商用前先核实许可",
        has_speaker=False, qc="none", needs_transcribe=True, quality=30,
    ),
    # ---- 中文（混合防遗忘，建议占比 10-20%） ----
    Source(
        "aishell3", "zh", "AISHELL-3（~85h 多说话人朗读）",
        "openslr", "https://www.openslr.org/resources/93/data_aishell3.tgz", "", "",
        "Apache-2.0", "约 20GB，下载耗时；有说话人列可 ref 配对，中文防遗忘首选",
        has_speaker=True, qc="none", role="antiforget", preferred=True, quality=90,
    ),
    Source(
        "fleurs_zh", "zh", "FLEURS 普通话（~10h，干净朗读）",
        "hf_dataset", "google/fleurs", "cmn_hans_cn", "train",
        "CC-BY-4.0", "少量高质补充",
        has_speaker=False, qc="none", role="antiforget", quality=80,
    ),
]


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(f"未知数据源: {source_id}，可选: {[s.id for s in SOURCES]}")


def row_passes(src: Source, get) -> bool:
    """已声明的过滤条件缺列/无效值时不放行，避免未知数据进入训练。"""
    for col, op, val in src.row_filters:
        v = get(col)
        if v is None:
            return False
        if op == "in" and v not in val:
            return False
        if op == "not_in" and v in val:
            return False
        if op in (">=", "<="):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(fv):
                return False
            if op == ">=" and fv < float(val):
                return False
            if op == "<=" and fv > float(val):
                return False
    return True


def source_id_from_display(label: str) -> str:
    """把下拉框的显示文本反解成数据源 id；已是纯 id 时原样返回。"""
    sid = (label or "").split(" — ")[0].replace(PREFERRED_TAG, "").strip()
    get_source(sid)  # 解析不出就直接抛，别把脏字符串带到下载环节
    return sid


def preferred_sources() -> dict[tuple[str, str], Source]:
    """每个 (语种, 角色) 下的首选源，用于页面标注与混合建议。"""
    return {(s.lang, s.role): s for s in SOURCES if s.preferred}


def sources_by_quality() -> list[Source]:
    """按语种分组，并在每组内按当前质量判断倒序排列。"""
    return sorted(SOURCES, key=lambda s: (s.lang, -s.quality, s.id))
