# 📺 YouTube Analyzer Pro

Transform any YouTube video into structured insights — summaries, key topics, and interactive Q&A — powered by LangChain, FAISS, and Groq's LLM API.

**Live demo:** [youtube-video-analyzer-pro-aryan-aman.streamlit.app](https://youtube-video-analyzer-pro-aryan-aman.streamlit.app)

---

## ✨ Features

- 🔗 Paste any YouTube URL and extract its transcript automatically
- 📊 AI-generated structured video summaries with timestamps
- 🔑 Key topic extraction from video content
- 💬 Interactive Q&A — ask anything about the video and get context-aware answers
- ⚡ Powered by FAISS vector search + Groq's fast LLM inference

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM Orchestration:** LangChain (LCEL)
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM Provider:** Groq (Llama 3.1)
- **Transcript Source:** `youtube-transcript-api`

---

## ⚠️ Known Limitation: Transcript Fetching on Cloud Deployment

This app fetches YouTube transcripts directly via `youtube-transcript-api`, which sends requests straight to YouTube's servers. **This works reliably when run locally**, but can intermittently fail when deployed on Streamlit Community Cloud.

**Why this happens:**
Streamlit Community Cloud runs on shared Google Cloud Platform infrastructure. YouTube actively rate-limits and blocks IP ranges associated with major cloud providers (AWS, GCP, Azure) to prevent automated scraping — this isn't specific to this app or its code, it affects any app fetching YouTube transcripts from a cloud IP.

**What this means in practice:**
- The app may return a "Could not retrieve a transcript" error for videos that work fine when run locally.
- This is **not a permanent, fixed block** — it fluctuates based on YouTube's anti-bot detection and which IPs in the shared cloud pool are currently flagged. It can work intermittently, fail for stretches of time, and resolve unpredictably. There is no fixed daily/monthly reset — it's traffic-pattern-based on YouTube's end, not a scheduled window.

**The standard fix (not implemented here by design):**
The reliable solution is routing transcript requests through a rotating residential proxy service (e.g. Webshare), as documented in the [`youtube-transcript-api` README](https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans). This project intentionally does not include proxy credentials to keep the deployment free, dependency-light, and credential-free for demo purposes.

**Recommended way to evaluate this project:**
Clone the repo and run it locally (`streamlit run app.py`) for guaranteed, reliable transcript fetching — the cloud demo link is provided for convenience but may occasionally show the IP-blocking error above depending on current conditions.

---

## 🚀 Running Locally

```bash
git clone https://github.com/aryanaman07/youtube-video-analyzer-pro.git
cd youtube-video-analyzer-pro
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

---

## 📄 License

This project is open source and available for educational/portfolio purposes.
