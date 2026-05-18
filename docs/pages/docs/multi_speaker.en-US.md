# Multi-Speaker Dubbing

Since v2.x VideoLingo can detect multiple speakers in a single video and assign each
speaker an independent TTS voice. This page covers the prerequisites, UI workflow,
three voice modes and common pitfalls.

---

## 1. Prerequisites

| Item | Requirement |
|---|---|
| ASR backend | **Soniox** or **ElevenLabs** (both ship diarization). `whisper_*` / `302_whisperX` are not supported and the UI hard-disables them. |
| Video length | The whole file is sent to ASR in one shot (no chunking). Soniox accepts ~5 h per request and the free $200 credit covers dozens of hours. |
| TTS backend | Any supported backend can act as a "fixed voice". Clone mode defaults to **GPT-SoVITS** but SiliconFlow CosyVoice2 / Fish Audio / mimo / F5-TTS are also supported. |

---

## 2. How to Enable

1. Set `multi_speaker_enabled: true` in `config.yaml` (or toggle it from the sidebar — both work).
2. Pick Soniox or ElevenLabs as ASR backend; make sure `soniox_diarize: true`.
3. The pipeline pauses at the **Speaker Preview** stage and lists a sample for each speaker:
   - Left: audio preview + a snippet of recognised text
   - Right: a 3-way dropdown (default / fixed / clone)
4. Click **Continue** once every speaker is decided. The pipeline resumes — no restart needed.

> 💡 Setting `multi_speaker_enabled` back to `false` makes the entire pipeline behave
> exactly as before with zero overhead.

---

## 3. The Three Voice Modes

### 1. `default`
Speaker_id is ignored; the global `tts_method` and its global voice are used.
Functionally identical to turning multi-speaker off.

### 2. `fixed`
Pin a voice name (e.g. `zh-CN-XiaoxiaoNeural`) for this speaker. Every clip from this
speaker uses that voice.
> Cross-backend is allowed: even if the global `tts_method` is azure, a speaker can be
> routed to edge_tts.

### 3. `clone`
Upload a reference clip (WAV / MP3, 3-10 s recommended). The pipeline uses GPT-SoVITS
(or the backend's own voice-clone path) to imitate the timbre.

Clone-capable backends:
- **gpt_sovits** (default)
- **sf_cosyvoice2** — feeds ref_wav directly, bypassing the `refers/<n>.wav` flow
- **sf_fishtts** — forces dynamic mode
- **mimo** — auto-switches to the `mimo-v2.5-tts-voiceclone` model
- **_302_f5tts** — caches the uploaded URL per ref_wav path (multi-speaker safe)

---

## 4. The `speaker_voice_map` Field

The UI writes this into `config.yaml`:

```yaml
speaker_voice_map:
  S1:
    mode: fixed
    voice: zh-CN-XiaoxiaoNeural
  S2:
    mode: clone
    ref_wav: D:/path/to/speaker2_ref.wav
  S3:
    mode: default
```

Hand-editing the yaml works too, but the UI validates that ref_wav exists and that
the voice id is legal — recommended.

---

## 5. Common Pitfalls

| Symptom | Cause / Fix |
|---|---|
| Only one voice in the final dub | Check that the ASR backend is Soniox/ElevenLabs; `whisper` returns empty speaker_ids. |
| Clone voice "leaks" between speakers | An older F5-TTS bug used a single global cache; this version buckets by ref_wav path. Open an issue if it resurfaces. |
| ref_wav file missing at runtime | The router auto-falls back to the global voice with a yellow warning — the pipeline keeps running. |
| Speaker IDs drift across segments | In "whole-file" mode Soniox is stable; if it still drifts, see Backlog → cross-segment voiceprint clustering. |

---

## 6. Rolling Back

Flip `multi_speaker_enabled` to `false` in `config.yaml` (or toggle it off in the UI).
- Leaving `speaker_voice_map` populated is fine — the router returns `None` on the
  very first line of `_multi_enabled()`.
- All TTS backend calls fall back to the global voice path with zero overhead.
