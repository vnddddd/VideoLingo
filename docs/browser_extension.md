# VideoLingo Chrome Extension

This project now includes a local Chrome extension workflow for YouTube:

1. The extension reads the current YouTube tab URL.
2. It exports current YouTube cookies from Chrome and sends them only to the local bridge.
3. The bridge downloads audio only, then runs VideoLingo transcription, translation and dubbing.
4. The bridge saves the finished dub/subtitle in a persistent local cache keyed by YouTube video ID.
5. The extension overlays the normalized plugin audio and `dub.srt` on the YouTube page, muting the original video audio.

## Start the Local Bridge

From the project root:

```bat
StartBrowserBridge.bat
```

Or:

```bat
.venv\Scripts\python.exe tools\browser_bridge.py
```

The bridge listens on:

```text
http://127.0.0.1:8765
```

It writes the exported YouTube cookie file to:

```text
cookies/browser_bridge_youtube.txt
```

and updates `youtube.cookies_path` in `config.yaml`.

## Load the Chrome Extension

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click "Load unpacked".
4. Select this folder:

```text
chrome_extension
```

## Use on YouTube

1. Open a YouTube video page.
2. Click the VideoLingo extension.
3. Click "Translate and Dub".
4. Keep the local bridge running while the job runs.
5. When done, the extension applies the translated dub/subtitle overlay automatically. If it does not, click "Apply Overlay".

After a video has completed once, opening the same YouTube video again will automatically reuse the cached dub and subtitle. Different URL forms for the same video, such as `watch?v=...`, `youtu.be/...`, and `shorts/...`, share the same cache key.

The generated files for extension jobs are saved under:

```text
runtime/browser_bridge/jobs/<job_id>/
```

Partial in-progress outputs are saved under:

```text
runtime/browser_bridge/jobs/<job_id>/work_output/
```

The persistent cache index is saved at:

```text
runtime/browser_bridge/index.json
```

## Notes

- The bridge accepts multiple jobs, but runs them one at a time because VideoLingo uses the shared `output/` workspace.
- You can watch one completed dubbed video while queueing another YouTube video for translation.
- The bridge restores each job's own `work_output/` into `output/` before running, so VideoLingo's existing checkpoint checks can skip completed steps after a restart.
- The original `dub.mp3` is kept, and the extension plays `dub_loudnorm.mp3` for browser playback.
- Cookies are written locally and are not sent anywhere except the local bridge on `127.0.0.1`.
