from __future__ import annotations

import math
from dataclasses import dataclass


PREFERRED_TAG = "【首选】"

# 微调目标语种。zh / en 是防遗忘回放语种，不是目标，别加进来。
# UI 文案与 eval 白名单一律引用这里，不要再手写一遍语种列表。
TARGET_LANGS = ("th", "tl", "vi", "id", "ms")

# Whisper 会把句内英文多的 code-switch 样本判成 en，只认目标语种会误杀最该保留的样本。
# id 的日常口语与短剧台词混英文程度接近 Taglish，ms（Manglish）同理；vi 的混英以词内
# 借词为主，默认从严，实测 drop_lang 误杀再给该源单独加 accept_langs=("vi", "en")。
CODE_SWITCH_ACCEPT = {"tl": ("tl", "en"), "id": ("id", "en"), "ms": ("ms", "en")}


@dataclass(frozen=True)
class Source:
    id: str
    lang: str  # th / tl / vi / id / ms / zh / en
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

        tl / id / ms 默认放行 en：短剧台词与自然口语都是 code-switch（Taglish、Manglish），
        句内英文词多的样本 Whisper 常判成 en，只认目标语种会把最该保留的样本全部误杀。
        vi 不在此列，实测误杀再按源加 accept_langs。
        """
        if self.accept_langs:
            return self.accept_langs
        return CODE_SWITCH_ACCEPT.get(self.lang, (self.lang,))

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
    Source("drama_vi", "vi", "已审核真人越南语短剧对白（自备 JSONL）", "local",
           license="按自有授权",
           note="越南语唯一的可信表现力来源：没有已核实的开源真人情感/表演语料。"
                "填写 speaker_verified=true；英文/中文同人参考可标 reference_only=true",
           has_speaker=True, role="expressive", preferred=True, expressive=True, quality=100),
    Source("drama_id", "id", "已审核真人印尼语短剧对白（自备 JSONL）", "local",
           license="按自有授权",
           note="印尼语唯一的可信表现力来源：没有已核实的开源真人情感/表演语料。"
                "台词混英文时转写会判成 en，本源默认放行 (id, en)",
           has_speaker=True, role="expressive", preferred=True, expressive=True, quality=100),
    Source("drama_ms", "ms", "已审核真人马来语短剧对白（自备 JSONL）", "local",
           license="按自有授权",
           note="马来语唯一的可信表现力来源：没有已核实的开源真人情感/表演语料。"
                "台词混英文（Manglish）时转写会判成 en，本源默认放行 (ms, en)。"
                "填写 speaker_verified=true；英文/中文同人参考可标 reference_only=true",
           has_speaker=True, role="expressive", preferred=True, expressive=True, quality=100),
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
        "未知", "待审计候选：缺文本/说话人/完整来源说明，核验真人录音和情绪标签后再使用。"
        "已核实数据形态：只有 audio+label 两列（无文本，全库 13811 条都要转写）；"
        "原始 wav 平均 151KB/条，按 16k/16bit 折算中位约 1.6s，多数低于 3s 时长下限，"
        "短情绪爆发过 VAD + 置信度门限几乎必被排除——先量时长分布再决定是否值得加工",
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
    # ---- 越南语 ----
    Source(
        "gigaspeech2_vi", "vi", "Gigaspeech 2 越南语（YouTube/播客自然口语）",
        "hf_dataset", "speechcolab/gigaspeech2", "vi", "train",
        "Apache-2.0",
        "许可最干净（Apache-2.0，无 SA/NC 红线），是 vi 的自然口语首选锚点。"
        "⚠️ 规模与字段形态本轮未核实：先 --max-samples 20 试跑，确认 audio 列存在且"
        "时长落在 3-30s；若是 path+start+end 的长音频切片形态，下载器会整段读入而非"
        "按时间戳切，本源即不可用（见 docs/vi_id_support.md）。"
        "gated:auto，需 .env 配 HF_TOKEN 并在数据集页面同意条款。无 speaker ID，不配 ref",
        has_speaker=False, qc="none", role="anchor", preferred=True, quality=70,
    ),
    Source(
        "fleurs_vi", "vi", "FLEURS 越南语（干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "vi_vn", "train",
        "CC-BY-4.0",
        "config 名按 FLEURS 的 {lang}_{country} 模式推得，首次先 --max-samples 50 确认；"
        "小时数未核实。朗读发音补充；无可靠说话人身份，不配 ref",
        has_speaker=False, qc="none", accept_langs=("vi",), quality=60,
    ),
    Source(
        "cv22_vi", "vi", "Common Voice 22 越南语（众包朗读，发音锚点）",
        "hf_dataset", "fsicoli/common_voice_22_0", "vi", "train",
        "CC0",
        "官方已撤架，此为社区镜像（无需同意条款）；该 locale 的 validated 小时数未核实，"
        "先 --max-samples 试跑。众包噪音大，自动 Whisper 校验；朗读语料有权威文本，"
        "语种不符就是错行，因此不吃 code-switch 放行",
        has_speaker=True, qc="whisper", accept_langs=("vi",), quality=30,
    ),
    # ---- 印尼语 ----
    Source(
        "gigaspeech2_id", "id", "Gigaspeech 2 印尼语（YouTube/播客自然口语）",
        "hf_dataset", "speechcolab/gigaspeech2", "id", "train",
        "Apache-2.0",
        "许可最干净（Apache-2.0，无 SA/NC 红线），是 id 的自然口语首选锚点。"
        "⚠️ 规模与字段形态本轮未核实：先 --max-samples 20 试跑，确认 audio 列存在且"
        "时长落在 3-30s；若是 path+start+end 的长音频切片形态，本源即不可用。"
        "gated:auto，需 HF_TOKEN 并在页面同意条款。无 speaker ID，不配 ref",
        has_speaker=False, qc="none", role="anchor", preferred=True, quality=70,
    ),
    Source(
        "fleurs_id", "id", "FLEURS 印尼语（干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "id_id", "train",
        "CC-BY-4.0",
        "config 名按 FLEURS 的 {lang}_{country} 模式推得，首次先 --max-samples 50 确认；"
        "小时数未核实。朗读发音补充；无可靠说话人身份，不配 ref",
        has_speaker=False, qc="none", accept_langs=("id",), quality=60,
    ),
    Source(
        "cv22_id", "id", "Common Voice 22 印尼语（众包朗读，发音锚点）",
        "hf_dataset", "fsicoli/common_voice_22_0", "id", "train",
        "CC0",
        "官方已撤架，此为社区镜像（无需同意条款）；该 locale 的 validated 小时数未核实，"
        "先 --max-samples 试跑。众包噪音大，自动 Whisper 校验；朗读语料有权威文本，"
        "语种不符就是错行，因此不吃 id 默认的 (id, en) 放行",
        has_speaker=True, qc="whisper", accept_langs=("id",), quality=30,
    ),
    # ---- 马来语 ----
    Source(
        "yodas2_ms", "ms", "YODAS2-Sidon 马来语（YouTube 自发口语，Sidon 降噪，24kHz）",
        "hf_dataset", "sarulab-speech/yodas2_sidon", "ms000", "train",
        "CC-BY-3.0",
        "ms 的自然口语首选：与已在用的 yodas_th 同一上游家族（YODAS2 + Sidon 降噪），"
        "许可干净（CC-BY-3.0，无 SA/NC 红线），config ms000 已由 parquet API 核实存在。"
        "⚠️ 仓库是 WebDataset .tar.gz 分片（flac + 可选 metadata.json），**规模与字段形态"
        "未核实**，先 --max-samples 20 试跑确认 audio 列存在且时长落 3-30s；若转换 parquet "
        "只有元数据、音频留在 .tar 里，本源即不可用。试跑后还要把 metadata.json 里的 "
        "YouTube video ID 映射到 session_col，否则同一视频的切片会跨 train/val 泄漏。"
        "YouTube 抓取，speaker 是视频级近似身份，不作 ref 依据。详见 docs/ms_support.md",
        has_speaker=False, qc="none", role="anchor", preferred=True, quality=80,
    ),
    Source(
        "fleurs_ms", "ms", "FLEURS 马来语（干净朗读，发音锚点）",
        "hf_dataset", "google/fleurs", "ms_my", "train",
        "CC-BY-4.0",
        "config 名是 `ms_my`——结尾的 my 是**国家码马来西亚**，不是缅甸语，别认错。"
        "config 已由 HF parquet API 核实存在（103 个 config 之一），小时数未核实，"
        "首次先 --max-samples 50 确认。朗读发音补充；无可靠说话人身份，不配 ref。"
        "有权威文本，语种不符就是错行，因此用 accept_langs 覆盖掉 ms 默认的 (ms, en) 放行",
        has_speaker=False, qc="none", accept_langs=("ms",), quality=60,
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
