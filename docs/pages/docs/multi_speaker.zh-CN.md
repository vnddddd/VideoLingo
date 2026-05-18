# 多说话人配音

VideoLingo 自 v2.x 起支持「同一视频里识别出多个说话人并为每位说话人指派独立音色」。
本文介绍启用条件、UI 操作流程、三种音色模式与常见坑。

---

## 一、前置条件

| 项 | 要求 |
|---|---|
| ASR 后端 | **Soniox** 或 **ElevenLabs**(均带 diarization)。`whisper_*` / `302_whisperX` 不支持,UI 会硬禁用。 |
| 视频长度 | 整段一次性送 ASR(无切片)。Soniox 单次上限约 5 小时,免费额度 $200 够跑数十小时。 |
| TTS 后端 | 任意支持的 backend 均可作为「固定音色」。clone 模式默认走 **GPT-SoVITS**,亦支持 SiliconFlow CosyVoice2 / Fish Audio / mimo / F5-TTS。 |

---

## 二、启用步骤

1. `config.yaml` 把 `multi_speaker_enabled` 改为 `true`(或者在 UI 侧边栏勾选,效果等价)。
2. ASR 后端选 Soniox 或 ElevenLabs;确保 `soniox_diarize: true`。
3. 跑到 **Speaker Preview** 阶段,UI 会暂停并列出每个 speaker 的样本片段:
   - 左侧:该 speaker 的音频试听 + 一段识别文本
   - 右侧:三选一下拉(default / fixed / clone)
4. 全部确认后点 **Continue**,pipeline 续跑无需重启。

> 💡 关掉 `multi_speaker_enabled` 之后整条 pipeline 行为与改造前**完全等同**,零开销。

---

## 三、三种音色模式

### 1. `default`
忽略 speaker,沿用全局 `tts_method` + 该 backend 的全局 voice。等同于关掉多说话人。

### 2. `fixed`
指定某 backend 的音色名(如 `zh-CN-XiaoxiaoNeural`)。该 speaker 的所有片段都用这个固定音色。
> 可跨 backend:全局 tts_method 是 azure,但某 speaker 可单独走 edge_tts。

### 3. `clone`
上传一段该 speaker 的参考音频(WAV / MP3,3-10 秒最佳),pipeline 用 GPT-SoVITS(或 backend 自带的 voice-clone 通道)模仿。

支持 clone 的 backend:
- **gpt_sovits**(默认)
- **sf_cosyvoice2** —— 直接喂 ref_wav,绕过 `refers/<n>.wav` 流程
- **sf_fishtts** —— 强制 dynamic 模式
- **mimo** —— 自动切到 `mimo-v2.5-tts-voiceclone` 模型
- **_302_f5tts** —— 多 speaker 时按 ref_wav 路径独立缓存上传 URL

---

## 四、`speaker_voice_map` 字段

UI 写入 `config.yaml`,结构如下:

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

手改 yaml 也行,但建议用 UI(自动校验 ref_wav 存在且 voice 合法)。

---

## 五、常见坑

| 现象 | 原因 / 处理 |
|---|---|
| 跑完发现只有一种音色 | 检查 ASR 后端是否选了 Soniox/ElevenLabs;`whisper` 输出 speaker_id 全空。 |
| clone 音色「串台」 | F5-TTS 旧版有全局 ref URL 缓存 bug,本版本已按 ref_wav 路径分桶,如复发请提 issue。 |
| ref_wav 文件丢失 | router 自动回退到全局 voice 并打 yellow warning,不会中断 pipeline。 |
| 跨片段同一 speaker 编号不一致 | Soniox 在「整段送」模式下编号稳定;若仍跨段乱,见 Backlog 跨段声纹聚类。 |

---

## 六、回退方法

直接把 `config.yaml` 的 `multi_speaker_enabled` 改回 `false`(或在 UI 关掉)即可。
- `speaker_voice_map` 保留也无影响,router 会在 `_multi_enabled()` 第一行返回 None。
- 所有 TTS backend 调用回到全局 voice 路径,无任何额外开销。
