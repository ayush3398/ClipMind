# ClipMind — AI YouTube Transcript Summarizer

Turn any YouTube video into clear, actionable insights: executive summary, key takeaways, quotes, action items, and a chat assistant grounded in the transcript. Powered by Claude.

**Live demo:** `https://your-app-name.streamlit.app` *(placeholder — replace after deploying)*

## Screenshots

| Home | Report | Chat |
| --- | --- | --- |
| *(screenshot placeholder)* | *(screenshot placeholder)* | *(screenshot placeholder)* |

## Features

- **Smart URL handling:** accepts `youtube.com/watch`, `youtu.be`, Shorts, embed, and live links, and validates them before doing any work.
- **Video overview:** title, channel, thumbnail, transcript word count, estimated reading time, and video length.
- **Structured AI report:** executive summary, 5–8 key takeaways, important quotes or insights, action items, topics covered, and a "Who should watch this?" recommendation.
- **Four summary styles:** Brief, Detailed, Study notes, Business insights.
- **Three output languages:** English, Hindi (Devanagari), and Hinglish (Hindi in Roman script).
- **Ask ClipMind:** a per-video chat that answers from the transcript, cites timestamps, and says so when the transcript doesn't cover a question.
- **Long-video support:** transcripts over ~14,000 words are split into chunks, summarized in parallel, then combined into a single report.
- **Downloads:** the full report as Markdown, plus the raw transcript as text.
- **Graceful errors:** invalid URLs, missing or disabled captions, private videos, blocked requests, API failures, rate limits, and overly long transcripts each get a clear message.
- **Premium UI:** dark glassmorphism theme, gradient accents, animated loading state, responsive layout.
- **Configurable model:** defaults to `claude-sonnet-4-5`; switch models or enter any model ID from the sidebar.

## Tech stack

| Layer | Technology |
| --- | --- |
| UI | Streamlit with custom CSS |
| Transcripts | `youtube-transcript-api` |
| LLM | Anthropic Claude via the official Python SDK |
| Config | Streamlit secrets, `python-dotenv` for local `.env` |
| Hosting | Streamlit Community Cloud |

## Architecture

```
 YouTube URL
     │
     ▼
 Validate + extract video ID ──► oEmbed (title, channel)
     │
     ▼
 youtube-transcript-api ──► timestamped ~40 s lines
     │
     ├── ≤ 14,000 words ───────────────► single Claude call
     │
     └── > 14,000 words ► split into ~9,000-word chunks
                          ► Claude summarizes each chunk (parallel)
                          ► Claude combines chunk notes into the final report
     │
     ▼
 Tagged Markdown sections ──► parsed into Summary / Key Takeaways tabs
     │
     ├── Markdown download
     └── Chat tab: transcript (or condensed notes for very long videos)
         + history sent to Claude; answers grounded in the transcript
```

Design notes:

- **Stateless server:** everything lives in `st.session_state`; no files or databases are written.
- **Structured output:** Claude returns XML-style tags (`<executive_summary>`, `<key_takeaways>`, …) containing Markdown. The app parses them into separate tabs and falls back to showing the raw text if parsing fails.
- **Hallucination control:** prompts require Claude to use only the provided material, flag unclear captions, and treat transcript text as data, not instructions.
- **Cost control:** the chat system prompt uses Anthropic prompt caching, so repeated questions about one video reuse the cached transcript.
- **Chat in a fragment:** the chat tab uses `st.fragment`, so sending a message doesn't reset the selected tab.

## Local setup

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/clipmind.git
cd clipmind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (already git-ignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Alternatively, create a `.env` file (also git-ignored) with `ANTHROPIC_API_KEY=sk-ant-...`.

Run the app:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to a **public GitHub repo**, with `app.py` and `requirements.txt` in the repository root.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo and branch, and set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**. Add the final URL to the **Live demo** line above.

## Secrets reference

| Key | Required | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes | Authenticates with the Claude API. |
| `YT_PROXY_URL` | No | HTTP(S) proxy URL used only for transcript requests. |
| `WEBSHARE_PROXY_USERNAME`, `WEBSHARE_PROXY_PASSWORD` | No | Webshare rotating residential proxy credentials (alternative to `YT_PROXY_URL`). |

The API key is never hard-coded and is read only from Streamlit secrets (or the environment for local use).

## Troubleshooting: "YouTube blocked the transcript request"

YouTube frequently blocks requests that come from cloud provider IP addresses, which includes Streamlit Community Cloud. It works reliably on a local machine but may fail intermittently once deployed. If that happens:

- Retry after a minute or try a different video.
- Configure a residential proxy through the optional secrets above. This is the dependable fix for production use.
- Record your demo from a local run if you need a guaranteed result for an interview.

## Disclaimer

Summaries depend on the captions available on YouTube. Auto-generated captions can contain errors, and videos without captions can't be summarized. AI output can be wrong or incomplete, so verify anything important against the original video. ClipMind is not affiliated with YouTube or Google.
