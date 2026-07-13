<div align="center">

# 🎙️ VOXDYNAMICS

### *Deep Emotion Extraction Layer*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-15-336791?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <br/>
  <img src="https://img.shields.io/badge/WebSocket-Enabled-6c47ff?style=flat-square" />
  <img src="https://img.shields.io/badge/CNN%20Accuracy-97.25%25-00ff87?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square" />
</p>

**Upload audio → AI segments utterances automatically → 5 interactive charts render in real-time**

⚡ **97.25% accuracy** · 🧠 Intelligent VAD Segmentation · 🎨 Real-time Visualizations

</div>

---

## 🎬 Live Demo

<p align="center">
  <img src="docs/images/mini_demo.gif" alt="VoxDynamics Demo" width="720" />
  <br/>
  <em>Upload → Analyze → 5 interactive charts in real-time</em>
</p>

---

## ✨ Features at a Glance

| | Feature | Description |
|:---:|:--------|:------------|
| 🎯 | **97.25% Accuracy** | Deep 1D-CNN trained on 48,648 samples (RAVDESS + CREMA-D) |
| 🧠 | **Smart VAD Segmentation** | Silero VAD detects speech islands, merges pauses, filters noise |
| 🎨 | **5 Interactive Charts** | Radar · Donut · Emotion Waveform · Confidence Stream · Segment Log |
| 🎧 | **WaveSurfer Player** | Playable waveform with seek sync — click chart → jump to segment |
| ⚡ | **Real-Time Ready** | WebSocket streaming for live emotion analysis from microphone |
| 🗄️ | **Session History** | Every analysis persisted to PostgreSQL — browse, reload, re-analyze |
| 🐳 | **One-Command Deploy** | `docker-compose up -d` — app + database in 3 minutes |
| 🌐 | **Supports All Audio** | `.wav`, `.mp3`, `.flac` — any length, any sample rate |

---

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 3 Commands to Launch

```bash
# 1. Clone
git clone <repository-url>
cd VoxDynamics

# 2. Configure (or use defaults)
cp .env.example .env

# 3. Launch everything
docker-compose up -d --build
```

**→ Open [http://localhost:8000](http://localhost:8000)** in your browser.

| Service | Port | Description |
|:--------|:----:|:------------|
| `voxdynamics-app` | 8000 | FastAPI + CNN inference engine |
| `voxdynamics-db` | 5432 | PostgreSQL 15 database |

> ⏱️ First startup takes ~30s while models download. Check status at `http://localhost:8000/health`.

---

## 🖼️ UI Showcase

### 1 · Upload & Preview

<p align="center">
  <img src="docs/images/home_page.png" alt="Home Page" width="700" />
  <br/>
  <em>Drag & drop audio files — WaveSurfer.js renders waveform instantly</em>
</p>

### 2 · Analysis Report

<p align="center">
  <img src="docs/images/analysis_screen.png" alt="Analysis Report" width="700" />
  <br/>
  <em>Dominant Emotion Card · Emotion Signature Radar · Emotion Waveform Analysis</em>
</p>

### 3 · Charts & Segment Log

<p align="center">
  <img src="docs/images/analysis_screen_2.png" alt="Analysis Screen 2" width="700" />
  <br/>
  <em>Emotion Distribution Donut · Confidence Stream · Micro-Segment Detection Log</em>
</p>

### 4 · Session History

<p align="center">
  <img src="docs/images/history_screen.png" alt="History Screen" width="700" />
  <br/>
  <em>Every analysis session persisted — browse, reload, explore past reports</em>
</p>

---

## 🎯 How It Works

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#0d0d1e', 'primaryTextColor': '#f0f0f8', 'primaryBorderColor': '#00e5ff', 'lineColor': '#00e5ff', 'secondaryColor': '#13132a', 'tertiaryColor': '#07070f'}}}%%
flowchart LR
    subgraph Input["🎤 INPUT"]
        A1["🎵 .wav / .mp3 / .flac"]
    end

    subgraph Backend["⚙️ AI PIPELINE"]
        B1["🌐 Global Normalization"]
        B2["🧠 Silero VAD<br/>16kHz Scan"]
        B3["📦 Speech Island<br/>Clustering"]
        B4["🔬 Feature Extraction<br/>ZCR + RMS + MFCC"]
        B5["🤖 1D-CNN<br/>97.25% Accuracy"]
        B6["💾 PostgreSQL<br/>Session Storage"]
    end

    subgraph Frontend["🎨 VISUALIZATION"]
        C1["📊 5 Plotly Charts"]
        C2["🎧 WaveSurfer Player"]
        C3["📋 Segment Log Table"]
    end

    A1 -->|"POST /api/analyze"| B1
    B1 --> B2
    B2 -->|"speech islands"| B3
    B3 -->|"+ 200ms buffer"| B4
    B4 -->|"2,376 features"| B5
    B5 -->|"JSON 200"| B6
    B5 --> C1
    B5 --> C2
    B5 --> C3

    style Input fill:#0d0d1e,stroke:#00e5ff,stroke-width:2px,color:#f0f0f8
    style Backend fill:#07070f,stroke:#00ff87,stroke-width:2px,color:#f0f0f8
    style Frontend fill:#0d0d1e,stroke:#ab47bc,stroke-width:2px,color:#f0f0f8
    style A1 fill:#13132a,stroke:#00e5ff,stroke-width:1px,color:#f0f0f8
    style B1 fill:#13132a,stroke:#00ff87,stroke-width:1px,color:#f0f0f8
    style B2 fill:#13132a,stroke:#00ff87,stroke-width:1px,color:#f0f0f8
    style B3 fill:#13132a,stroke:#00ff87,stroke-width:1px,color:#f0f0f8
    style B4 fill:#13132a,stroke:#00ff87,stroke-width:1px,color:#f0f0f8
    style B5 fill:#13132a,stroke:#00ff87,stroke-width:1px,color:#f0f0f8
    style B6 fill:#13132a,stroke:#f9a825,stroke-width:1px,color:#f0f0f8
    style C1 fill:#13132a,stroke:#ab47bc,stroke-width:1px,color:#f0f0f8
    style C2 fill:#13132a,stroke:#ab47bc,stroke-width:1px,color:#f0f0f8
    style C3 fill:#13132a,stroke:#ab47bc,stroke-width:1px,color:#f0f0f8
```

**Detailed Flow:** Audio is globally normalized → VAD scans at 16kHz → Speech islands clustered (merging pauses < 0.8s) → Each segment + 200ms buffer → Features extracted (ZCR + RMS + MFCC = 2,376 values) → 1D-CNN predicts 1 of 7 emotions → Results stored in PostgreSQL → 5 interactive Plotly.js charts render.

---

## 🎧 Try the Sample

A multi-emotion validation file is included:

> **Angry → Happy → Surprised → Disgust → Sad (low-intensity)**

```bash
# The file should be at: data/emotions/mix/angry_happy_surprised_disgust_sad.wav
# If missing, you can create a test file by concatenating any audio samples
```

<!-- Audio element removed — sample file not yet committed to repo -->

---

## 🧠 Emotion Classes

| Emotion | Emoji | Color | Typical Confidence |
|:--------|:-----:|:-----:|:------------------:|
| Happy | 😊 | `#00ff87` | 95-100% |
| Angry | 😠 | `#ff416c` | 95-100% |
| Sad | 😢 | `#667eea` | 90-98% |
| Neutral | 😐 | `#9aa0b8` | 85-95% |
| Fearful | 😨 | `#f9a825` | 80-95% |
| Surprised | 😲 | `#ab47bc` | 80-95% |
| Disgust | 🤢 | `#ff8c00` | 85-97% |

---

## 📊 Performance

| Metric | Value |
|:-------|:-----:|
| **CNN Benchmark Accuracy** | **97.25%** (RAVDESS/CREMA-D) |
| **Real-World Accuracy** | **80%** (5-emotion mix file) |
| **Avg Confidence** | **96.5%** (on correct segments) |
| **Inference Speed** | ~0.26s / segment (CPU) |
| **Supported Emotions** | 7 discrete classes |
| **Min Audio Duration** | Any (VAD auto-detects) |
| **Max Upload Size** | 50 MB |
| **Supported Formats** | `.wav`, `.mp3`, `.flac` |

---

## 📁 Project Structure

```
VoxDynamics/
├── app/           # ★ Backend (FastAPI + AI pipeline + DB)
├── docs/          # 📖 Documentation
├── models/        # Pre-trained CNN weights + scaler
├── src/           # Research & training scripts + data loader
├── docker-compose.yml
└── Dockerfile
```

---

## 📖 Documentation

| Doc | Description |
|:----|:------------|
| [📐 Architecture](docs/ARCHITECTURE.md) | System design, AI pipeline, component deep-dive |
| [🔌 API Reference](docs/API.md) | Full API docs with request/response examples |
| [🛠️ Setup Guide](docs/SETUP.md) | Local dev setup, troubleshooting, configuration |
| [🚀 Deployment](docs/DEPLOYMENT.md) | Production deployment, security, monitoring |
| [📏 METHOD.md](docs/METHOD.md) | Full preprocessing methodology & rationale |
| [🧪 CNN Report](docs/benchmark/CNN_REPORT.md) | 6 experiments from 23% → 97.25% accuracy |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Backend** | FastAPI (Python) | REST API + WebSocket |
| **AI — VAD** | Silero VAD v4 (PyTorch) | Voice Activity Detection |
| **AI — CNN** | TensorFlow 2.x / Keras | Emotion classification (97.25%) |
| **Audio DSP** | librosa, soundfile, numpy | Feature extraction |
| **Database** | PostgreSQL 15 | Session persistence |
| **Frontend** | Vanilla HTML/CSS/JS | Dark glassmorphism SPA |
| **Charts** | Plotly.js | 5 interactive visualizations |
| **Audio Player** | WaveSurfer.js | Animated waveform player |
| **Container** | Docker + Compose | One-command deploy |

---

## 🤝 Contributing

We welcome contributions! Check out the [docs](docs/) to understand the system, then:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with** ❤️ **using FastAPI · TensorFlow · PostgreSQL · Docker**

<p>
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/API.md">API Docs</a> ·
  <a href="docs/SETUP.md">Setup</a> ·
  <a href="docs/DEPLOYMENT.md">Deploy</a>
</p>

<sub>⭐ Star us on GitHub — it helps!</sub>

</div>
