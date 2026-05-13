"""
Soniox 异步 ASR 响应适配器 (plan Task 2)
把 Soniox 的 subword-token 流转成 VideoLingo 下游期望的 whisper 风格 {"segments": [...]}

设计:
  Stage A: soniox_tokens_to_words(tokens)  -- subword 合并成 word
  Stage B: words_to_segments(words)        -- word 按 gap/speaker 切 segment (借鉴 elev2whisper)
  外壳   : soniox_to_whisper(api_resp)     -- 端到端,直接给上游用

Token 边界规则 (依据 smoke_transcript.json 实测):
  * token.text 以空格开头 -> 新 word 开始
  * 首 token 无前导空格也算新 word
  * speaker 变化 -> 强制新 word
  * is_audio_event=True 的 token 跳过(避免污染文本)
  * 标点/数字片段(无前导空格) -> 黏到前一 word

Segment 切分规则:
  * 与前一 word 的 gap > SPLIT_GAP_SEC -> 新段
  * speaker 变化 -> 新段

⚠ 下游 schema 契约 (core/asr_backend/audio_preprocess.py::process_transcription):
  * 每个 segment 必须含 "words" 列表 (即使开 word_level_timestamp=False 仍然依赖)
  * 每个 word 字典 key 必须是 "word" (不是 "text"!) + "start" + "end"
  * segment["speaker_id"] 可选
  (注: elev2whisper 输出 "text" 实为 VideoLingo 项目自身遗留 bug,
        本适配器以下游真实期望为准, 全模块统一用 "word" 字段)
"""
from typing import List, Dict, Any, Optional

SPLIT_GAP_SEC = 1.0  # 与 elevenlabs_asr.SPLIT_GAP 对齐


def soniox_tokens_to_words(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """subword tokens -> word 列表.
    每个输出 word: {word, start, end, speaker_id, language}
    start/end 单位: 秒 (float)
    """
    words: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for t in tokens:
        # 过滤音频事件 token (例如 <silence>, <laughter>),
        # 因为它们不属于文本流, 留给下游可单独消费
        if t.get("is_audio_event"):
            continue

        txt = t.get("text", "")
        if txt == "":
            continue

        spk = t.get("speaker")
        starts_new = (
            cur is None
            or txt.startswith(" ")
            or (spk is not None and cur.get("speaker_id") != spk)
        )

        if starts_new:
            if cur is not None:
                words.append(cur)
            cur = {
                "word": txt.lstrip(),  # 去除前导空格
                "start": t["start_ms"] / 1000.0,
                "end":   t["end_ms"]   / 1000.0,
                "speaker_id": spk,
                "language":   t.get("language"),
            }
        else:
            # 黏到当前 word: 标点/数字/subword 续接
            cur["word"] += txt  # 不 strip, subword 本身不含前导空格
            cur["end"]   = t["end_ms"] / 1000.0
            # speaker/language 若之前是 None, 用新值补
            if cur["speaker_id"] is None and spk is not None:
                cur["speaker_id"] = spk
            if cur["language"] is None and t.get("language") is not None:
                cur["language"] = t["language"]

    if cur is not None:
        words.append(cur)
    return words


def words_to_segments(words: List[Dict[str, Any]],
                      word_level_timestamp: bool = True,
                      split_gap_sec: float = SPLIT_GAP_SEC) -> List[Dict[str, Any]]:
    """word 列表 -> segment 列表 (whisper 风格).
    断段条件: 与下一 word 的 gap > split_gap_sec 或 speaker_id 变化 或 末尾.
    """
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    seg = {
        "text": "",
        "start": words[0]["start"],
        "end": words[0]["end"],
        "speaker_id": words[0].get("speaker_id"),
        "words": [],
    }

    for prev, nxt in zip(words, words[1:] + [None]):
        # 拼接 segment-level text (用单空格 join, 与 ElevenLabs 适配一致)
        if seg["text"] == "":
            seg["text"] = prev["word"]
        else:
            seg["text"] += " " + prev["word"]
        seg["end"] = prev["end"]
        if word_level_timestamp:
            seg["words"].append({
                "word":  prev["word"],
                "start": prev["start"],
                "end":   prev["end"],
            })

        # 判断是否在此处切段
        should_break = (nxt is None)
        if not should_break:
            gap = nxt["start"] - prev["end"]
            spk_change = (
                nxt.get("speaker_id") is not None
                and seg["speaker_id"] is not None
                and nxt.get("speaker_id") != seg["speaker_id"]
            )
            should_break = (gap > split_gap_sec) or spk_change

        if should_break:
            seg["text"] = seg["text"].strip()
            if not word_level_timestamp:
                seg.pop("words")
            segments.append(seg)
            if nxt is not None:
                seg = {
                    "text": "",
                    "start": nxt["start"],
                    "end": nxt["end"],
                    "speaker_id": nxt.get("speaker_id"),
                    "words": [],
                }
    return segments


def soniox_to_whisper(api_resp: Dict[str, Any],
                      word_level_timestamp: bool = True,
                      time_offset: float = 0.0) -> Dict[str, Any]:
    """完整适配: Soniox /v1/transcriptions/{id}/transcript 响应 -> {"segments": [...]}.
    time_offset: 秒, 用于切片转写后把 word 时间还原到全局.
    """
    tokens = api_resp.get("tokens", []) or []
    words = soniox_tokens_to_words(tokens)

    if time_offset:
        for w in words:
            w["start"] += time_offset
            w["end"]   += time_offset

    segments = words_to_segments(words, word_level_timestamp=word_level_timestamp)
    return {"segments": segments}


# --- 仅用于自检 / sanity 重建 ---
def reconstruct_text_from_tokens(tokens: List[Dict[str, Any]]) -> str:
    """直接把 token.text 全拼接并 strip, 模拟 API 端 text 字段的生成方式."""
    return "".join(t.get("text", "") for t in tokens if not t.get("is_audio_event")).strip()
