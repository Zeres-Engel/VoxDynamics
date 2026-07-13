# 🔌 API Reference

> **Complete API documentation for the VoxDynamics backend.**
> Base URL: `http://localhost:8000` (default)

---

## 📋 Table of Contents

- [Health Check](#-health-check)
- [Analyze Audio](#-analyze-audio)
- [List Sessions](#-list-sessions)
- [Get Session Details](#-get-session-details)
- [WebSocket Streaming](#-websocket-streaming)
- [Error Codes](#-error-codes)

---

## ✅ Health Check

Check if the server is running and AI models are loaded.

```
GET /health
```

### Response

```json
{
  "status": "ok",
  "models_loaded": true,
  "timestamp": "2026-07-13T14:30:00.123456",
  "version": "2.0.0"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` if server is running |
| `models_loaded` | boolean | `true` when VAD + CNN models are ready |
| `timestamp` | string (ISO) | Current server time |
| `version` | string | API version |

### Status Codes

| Code | Description |
|:----:|:------------|
| 200 | Server is healthy |
| 503 | Models still loading (during startup) |

---

## 🎙️ Analyze Audio

Upload an audio file for full emotion analysis. The file goes through the complete pipeline: VAD segmentation → CNN inference → DB persistence.

```
POST /api/analyze
```

### Request

| Parameter | Type | Required | Description |
|-----------|------|:--------:|:------------|
| `file` | UploadFile | ✅ | Audio file (.wav, .mp3, .flac) — max 50MB |

### Response

```json
{
  "session_uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "summary": {
    "dominant_emotion": "happy",
    "dominant_emoji": "😊",
    "avg_arousal": 0.7234,
    "avg_dominance": 0.6543,
    "avg_valence": 0.5123,
    "avg_scores": {
      "angry": 0.05,
      "disgust": 0.02,
      "fearful": 0.03,
      "happy": 0.72,
      "neutral": 0.08,
      "sad": 0.05,
      "surprised": 0.05
    },
    "avg_confidence": 0.9650,
    "audio_duration_s": 21.4,
    "speech_segments": 5
  },
  "segments": [
    {
      "time_s": 1.0,
      "duration_s": 2.0,
      "emotion_label": "angry",
      "arousal": 0.85,
      "dominance": 0.72,
      "valence": 0.15,
      "confidence": 0.97,
      "emoji": "😠",
      "color": "#FF4444",
      "scores": { ... },
      "is_speech": true
    },
    {
      "time_s": 3.5,
      "duration_s": 1.2,
      "emotion_label": "silence",
      "confidence": 0.0,
      "is_speech": false
    }
  ]
}
```

### Response Fields

#### `summary`

| Field | Type | Description |
|-------|------|-------------|
| `dominant_emotion` | string | Most frequent emotion across all segments |
| `dominant_emoji` | string | Emoji representation of dominant emotion |
| `avg_arousal` | float | Mean arousal (0.0–1.0) |
| `avg_dominance` | float | Mean dominance (0.0–1.0) |
| `avg_valence` | float | Mean valence (0.0–1.0) |
| `avg_scores` | object | Mean probability for each of the 7 emotions |
| `avg_confidence` | float | Mean prediction confidence |
| `audio_duration_s` | float | Total audio duration in seconds |
| `speech_segments` | int | Number of speech segments detected |

#### `segments[]` (speech)

| Field | Type | Description |
|-------|------|-------------|
| `time_s` | float | Start time in seconds |
| `duration_s` | float | Duration in seconds |
| `emotion_label` | string | Detected emotion label |
| `arousal` | float | Arousal (0.0–1.0) |
| `dominance` | float | Dominance (0.0–1.0) |
| `valence` | float | Valence (0.0–1.0) |
| `confidence` | float | Prediction confidence (0.0–1.0) |
| `emoji` | string | Emoji for the emotion |
| `color` | string | Hex color for the emotion |
| `scores` | object | Full probability distribution across 7 classes |
| `is_speech` | boolean | Always `true` for speech segments |

#### `segments[]` (silence)

| Field | Type | Description |
|-------|------|-------------|
| `time_s` | float | Start time |
| `duration_s` | float | Silence duration |
| `emotion_label` | string | `"silence"` |
| `is_speech` | boolean | `false` |

### Status Codes

| Code | Description |
|:----:|:------------|
| 200 | Analysis complete |
| 400 | Invalid or corrupted audio file |
| 422 | No speech detected in file |
| 503 | AI models still loading |

---

## 📜 List Sessions

Retrieve all historical analysis sessions with aggregated metadata.

```
GET /api/sessions
```

### Response

```json
{
  "sessions": [
    {
      "UUID": "a1b2c3d4-...",
      "Time": "07/13 14:30",
      "Dur.": "15s",
      "Points": 8,
      "A": 0.72,
      "D": 0.65,
      "V": 0.51
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `UUID` | string | Session UUID |
| `Time` | string | Formatted date/time (MM/DD HH:MM) |
| `Dur.` | string | Session duration (e.g. "15s") |
| `Points` | int | Number of emotion segments logged |
| `A` | float | Average arousal |
| `D` | float | Average dominance |
| `V` | float | Average valence |

### Status Codes

| Code | Description |
|:----:|:------------|
| 200 | Session list retrieved |
| 500 | Database connection error |

---

## 🔍 Get Session Details

Retrieve detailed emotion segment data for a specific session.

```
GET /api/emotions/{session_uuid}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|:--------:|:------------|
| `session_uuid` | string (path) | ✅ | UUID of the session |
| `limit` | int (query) | ❌ | Max records to return (default: 100, max: 1000) |

### Response

```json
{
  "session_uuid": "a1b2c3d4-...",
  "count": 8,
  "data": [
    {
      "id": 42,
      "session_id": 1,
      "timestamp": "2026-07-13T14:30:15",
      "emotion_label": "happy",
      "arousal": 0.7234,
      "dominance": 0.6543,
      "valence": 0.5123,
      "confidence": 0.9650,
      "time_s": 1.02,
      "duration": 2.05,
      "scores": { ... },
      "latency_ms": 256.78
    }
  ]
}
```

### Status Codes

| Code | Description |
|:----:|:------------|
| 200 | Session data retrieved |
| 404 | Session UUID not found |
| 500 | Database error |

---

## 🔗 WebSocket Streaming

Real-time audio streaming for live emotion analysis.

```
WS /stream
```

### Protocol

| Direction | Format | Description |
|:---------:|:-------|:------------|
| Client → Server | Binary | PCM float32 audio frames, 16kHz, mono |
| Server → Client | JSON | Structured emotion prediction result |

### Client → Server

Send raw audio bytes as binary frames:

```javascript
const ws = new WebSocket('ws://localhost:8000/stream');

// Record from microphone or stream audio
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const recorder = new AudioContext({ sampleRate: 16000 });
    // ... encode as PCM float32, 16kHz mono
    // Send chunks:
    ws.send(audioChunkBytes);
  });
```

### Server → Client

```json
{
  "emotion_label": "happy",
  "arousal": 0.85,
  "dominance": 0.72,
  "valence": 0.65,
  "confidence": 0.97,
  "emoji": "😊",
  "color": "#FFD700",
  "scores": {
    "happy": 0.97,
    "neutral": 0.02,
    "angry": 0.01
  },
  "is_speech": true,
  "latency_ms": 145.32,
  "session_id": "a1b2c3d4",
  "buffer_seconds": 2.5
}
```

### WebSocket Events

| Event | Action |
|-------|--------|
| Connection | Server accepts and waits for binary frames |
| Binary message | Process frame through pipeline, return JSON result |
| Disconnect | Client disconnected (session logged if speech detected) |
| Error | Server closes with code 1011 |

---

## ⚠️ Error Codes

| HTTP Code | Meaning | Common Causes |
|:---------:|:--------|:--------------|
| 400 | Bad Request | Corrupt audio file, unsupported format |
| 422 | Unprocessable Entity | No speech detected in upload |
| 503 | Service Unavailable | Models still loading at startup |
| 500 | Internal Error | Server or database failure |

### Error Response Format

```json
{
  "detail": "No speech detected in audio file."
}
```

---

## 💡 Rate Limits

| Limit | Value |
|:------|:-----:|
| Max file size | 50 MB |
| Max segments per response | 100 (use `limit` param) |
| Max session history | 50 most recent sessions |
| Concurrent WebSocket connections | Limited by server resources |

---

## 📦 Postman Collection

Import this collection to test the API:

```json
{
  "info": {
    "name": "VoxDynamics API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/health"
      }
    },
    {
      "name": "Analyze Audio",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/api/analyze",
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "file",
              "type": "file",
              "src": "/path/to/audio.wav"
            }
          ]
        }
      }
    },
    {
      "name": "List Sessions",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/api/sessions"
      }
    }
  ]
}
```

---

*For frontend integration examples, see the [app.js](/app/frontend/static/js/app.js) source code.*
