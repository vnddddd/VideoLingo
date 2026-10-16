# 多说话人配音

VideoLingo 自 v2.x 起支持「同一视频里识别出多个说话人并为每位说话人指派独立音色」。
本文介绍启用条件、UI 操作流程、三种音色模式与常见坑。

---

## 一、前置条件

| 项 | 要求 |
|---|---|
| ASR 后端 | **Soniox** 或 **ElevenLabs**(均带 diarization)。`whisper_*` / `302_whisperX` 不支持,UI 会硬禁用。 |
| 视频长度 | 整段一次性送 ASR(无切片)。Soniox 单次上限约 5 小时,免费额度 $200 够跑数十小时。 |
| TTS 后端 | clone 模式跟随全局 `tts_method`，需选择支持按说话人克隆的引擎，如 Fish Audio、Soniox。 |

---

## 二、启用步骤

1. `config.yaml` 把 `multi_speaker_enabled` 改为 `true`(或者在 UI 侧边栏勾选,效果等价)。
2. ASR 后端选 Soniox 或 ElevenLabs;确保 `soniox_diarize: true`。
3. 配音设置里选一个**克隆模型**(支持克隆的引擎)。此时一切自动完成,点一次「翻译并配音」即可跑到最后输出。
4. 若配音设置选的是**预置音色**(如 Fish/Soniox 的 preset、MiMo 的非克隆模型),则跑到 **Speaker Preview** 阶段会暂停并列出每个 speaker 的样本片段:
   - 左侧:该 speaker 的音频试听 + 一段识别文本
   - 右侧:三选一下拉(default / fixed / clone)
   全部确认后点 **Continue**,pipeline 续跑无需重启。

### 自动克隆(免确认)

当全局配音引擎本身就是克隆模式时,每位说话人都用**自己的预览音频做参考**、跟随全局模型克隆,因此没有任何需要用户决定的事情,UI 不再弹窗确认。
触发条件(与全局设置一致):

| 引擎 | 触发自动克隆的设置 |
|---|---|
| `fish_audio_tts` | `mode: custom` 或 `dynamic`(且模型不是仅付费、不支持多说话人的 `s1`) |
| `soniox_tts` | `mode: clone` |
| `sf_fish_tts` | `mode: custom` 或 `dynamic` |
| `mimo_tts` | 模型为 `mimo-v2.5-tts-voiceclone` |
| `gpt_sovits` / `sf_cosyvoice2` / `f5tts` / `indextts2` | 选中即克隆 |

不支持的引擎(如 edge_tts、qwen3_tts)和预设音色模式仍然保留手动选择。
仅当说话人不足 5 个词、没有可用音频片段时才会回退到全局音色。

发言少于 5 个词或没有可用音频片段的说话人不会出现在预览列表中。保存时会自动为这些说话人选择全局默认声音，其他说话人的独立音色选择不受影响。

> 💡 关掉 `multi_speaker_enabled` 之后整条 pipeline 行为与改造前**完全等同**,零开销。

---

## 三、三种音色模式

### 1. `default`
忽略 speaker,沿用全局 `tts_method` + 该 backend 的全局 voice。等同于关掉多说话人。

### 2. `fixed`
指定某 backend 的音色名(如 `zh-CN-XiaoxiaoNeural`)。该 speaker 的所有片段都用这个固定音色。
> 可跨 backend:全局 tts_method 是 azure,但某 speaker 可单独走 edge_tts。

### 3. `clone`
使用该说话人的预览音频作为参考，通过全局配音设置选定的引擎克隆。每位说话人使用自己的参考音频。

例如，全局选择 Fish Audio 的 `s2.1-pro-free`，所有人的克隆配音都使用该模型，但分别创建或复用各自的音色。Fish 的 `custom` / `dynamic` 克隆方式也跟随全局设置；全局为 `preset` 时，单独选择克隆的说话人使用 `custom`。不支持克隆的引擎会明确报错。

支持 clone 的 backend:
- **fish_audio_tts** —— 跟随全局 Fish 模型，每位说话人使用独立参考音频
- **soniox_tts** —— 用 ref_wav 创建/复用云端克隆音色,参考片段超过 18 秒会自动裁剪
- **gpt_sovits**
- **sf_cosyvoice2** —— 直接喂 ref_wav,绕过 `refers/<n>.wav` 流程
- **sf_fish_tts** —— 强制 dynamic 模式
- **mimo_tts** —— 自动切到 `mimo-v2.5-tts-voiceclone` 模型
- **f5tts** —— 多 speaker 时按 ref_wav 路径独立缓存上传 URL
- **indextts2** —— 使用该说话人的参考音频

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
