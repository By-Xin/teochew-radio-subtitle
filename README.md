# Teochew Radio Subtitle

准实时潮汕语电台字幕与普通话翻译原型。

当前目标不是重新训练模型，而是先把第一个可验证 milestone 跑通：

1. 从 TingFM 电台页面解析真实音频流。
2. 录制短音频片段并转为 16 kHz mono WAV。
3. 使用现有潮汕语 ASR 模型输出潮汕正字字幕。
4. 可选接入 OpenAI-compatible chat-completions 接口，把潮汕字幕翻译成普通话字幕。

## 当前基座

默认 ASR 模型：

```text
panlr/whisper-finetune-teochew
```

这个模型基于 OpenAI `whisper-medium` 微调，主要面向潮汕话正字识别。它不是普通话翻译模型，普通话翻译由后处理文本模型完成。

## 安装

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 解析电台流

默认目标是：

```text
https://tingfm.com/radio/16543?lang=zh_CN
```

运行：

```powershell
python scripts/radio_subtitle_pipeline.py resolve
```

## 录制音频

```powershell
python scripts/radio_subtitle_pipeline.py record --seconds 30 --output outputs/radio_sample.mp3
```

## 端到端字幕

```powershell
python scripts/radio_subtitle_pipeline.py run --seconds 30 --output-dir outputs/radio_16543
```

如果电台里有音乐或静音，建议先打开轻量 VAD：

```powershell
python scripts/radio_subtitle_pipeline.py run --seconds 30 --vad --output-dir outputs/radio_16543_vad
```

输出文件：

- `stream.json`: 电台与流信息
- `radio_sample.mp3`: 原始录制音频
- `radio_sample.wav`: ASR 输入音频
- `asr_raw.json`: 模型原始输出
- `segments.json`: 归一化字幕段
- `teochew.srt`: 潮汕语字幕
- `vad_segments.json`: 开启 `--vad` 时的人声候选片段

## 本地音频输入

用本地音频或视频文件验证模型效果，比直接监听音乐电台更可靠：

```powershell
python scripts/radio_subtitle_pipeline.py file `
  --input D:\path\to\sample.mp3 `
  --output-dir outputs/local_sample `
  --vad
```

只测试前 60 秒：

```powershell
python scripts/radio_subtitle_pipeline.py file `
  --input D:\path\to\sample.mp3 `
  --max-seconds 60 `
  --output-dir outputs/local_sample_60s `
  --vad
```

## 可选普通话翻译

配置 OpenAI-compatible API key：

```powershell
$env:OPENAI_API_KEY = "..."
python scripts/radio_subtitle_pipeline.py run --seconds 30 --translate --output-dir outputs/radio_16543_translated
```

使用其它兼容端点：

```powershell
python scripts/radio_subtitle_pipeline.py run `
  --seconds 30 `
  --translate `
  --translate-api-base "https://your-endpoint.example/v1" `
  --translate-model "your-model-name"
```

## Milestone

- M1: 离线短片段字幕链路，已跑通。
- M2: 支持本地音频文件输入，已实现。
- M3: 加入轻量 VAD/静音过滤，已实现第一版；音乐段过滤仍需加强。
- M4: 准实时监听模式，按 5-10 秒窗口持续输出字幕。
- M5: Web UI 或本地服务化部署。

## 注意

TingFM 和电台音频内容的版权归其原始权利方所有。本项目只提供技术原型，不附带任何音频数据或模型训练数据。

现有 `panlr/whisper-finetune-teochew` 和相关数据集授权需要单独核查，尤其是商业使用场景。
