# ============================================================
# VoxDynamics — Gradio Dashboard
# ============================================================
"""
Real-time Speech Emotion Recognition Dashboard.

Features:
  - Live microphone streaming
  - Current emotion display (label + emoji + confidence)
  - Arousal / Dominance / Valence gauges
  - Emotion timeline chart (Plotly)
  - Latency indicator
  - Session history
"""

import time
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from collections import deque
from typing import Optional

from app.core.processor import AudioProcessor
from app.core.emotion_model import EMOTION_COLORS, EMOTION_EMOJI


# ── Timeline buffer for charts ──────────────────────────────
MAX_TIMELINE_POINTS = 150


def _create_empty_timeline_chart() -> go.Figure:
    """Create an empty Plotly timeline chart."""
    fig = go.Figure()

    for dim, color in [
        ("Arousal", "#FF6B6B"),
        ("Dominance", "#4ECDC4"),
        ("Valence", "#45B7D1"),
    ]:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines+markers",
            name=dim, line=dict(color=color, width=2),
            marker=dict(size=3),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=300,
        title=dict(
            text="Emotion Dimensions Timeline",
            font=dict(size=14, color="#aaa"),
        ),
        xaxis=dict(
            title="Time (s)",
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Value",
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        showlegend=True,
    )
    return fig


def _create_emotion_bar_chart(result: dict) -> go.Figure:
    """Create a horizontal bar chart showing emotion dimensions."""
    dims = ["Arousal", "Dominance", "Valence"]
    vals = [result.get("arousal", 0), result.get("dominance", 0), result.get("valence", 0)]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dims, x=vals,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.3)", width=1),
        ),
        text=[f"{v:.2f}" for v in vals],
        textposition="inside",
        textfont=dict(size=14, color="white"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,0.8)",
        margin=dict(l=80, r=20, t=10, b=10),
        height=150,
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(tickfont=dict(size=13, color="#ccc")),
        showlegend=False,
    )
    return fig


def create_dashboard(processor: AudioProcessor) -> gr.Blocks:
    """Build and return the Gradio Blocks dashboard."""

    # State containers
    timeline_arousal = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_dominance = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_valence = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_times = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_labels = deque(maxlen=MAX_TIMELINE_POINTS)
    start_time = [time.time()]

    # ── Custom CSS ───────────────────────────────────────────
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .gradio-container {
        font-family: 'Inter', sans-serif !important;
        max-width: 1400px !important;
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    }

    .main-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95em;
        margin-bottom: 20px;
    }

    .emotion-display {
        text-align: center;
        padding: 25px;
        border-radius: 16px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }

    .status-card {
        padding: 15px;
        border-radius: 12px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }

    .metric-value {
        font-size: 1.8em;
        font-weight: 600;
        color: #fff;
    }

    .metric-label {
        font-size: 0.85em;
        color: #888;
        margin-top: 4px;
    }
    """

    # ── Build UI ─────────────────────────────────────────────
    with gr.Blocks(
        title="VoxDynamics — Real-Time SER",
        theme=gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#f0f0ff", c100="#e0e0ff", c200="#c0c0ff",
                c300="#a0a0ff", c400="#8080ff", c500="#667eea",
                c600="#5050dd", c700="#4040cc", c800="#3030aa",
                c900="#202088", c950="#101066",
            ),
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=custom_css,
    ) as demo:

        # Header
        gr.HTML("""
            <div class="main-title">🎙️ VoxDynamics</div>
            <div class="subtitle">Real-Time Speech Emotion Recognition • Language-Agnostic • Powered by Wav2Vec2</div>
        """)

        with gr.Row():
            # ── Left Column: Controls & Status ───────────────
            with gr.Column(scale=1):
                gr.HTML('<h3 style="color:#aaa; margin-bottom:10px;">🎤 Audio Input</h3>')

                audio_input = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone",
                    elem_id="mic-input",
                )

                gr.HTML('<h3 style="color:#aaa; margin-top:20px;">📊 Dimensions</h3>')
                dimension_chart = gr.Plot(
                    value=_create_emotion_bar_chart(
                        {"arousal": 0.0, "dominance": 0.0, "valence": 0.0}
                    ),
                    label="Emotion Dimensions",
                )

                with gr.Row():
                    latency_display = gr.Textbox(
                        label="⚡ Latency",
                        value="-- ms",
                        interactive=False,
                        elem_classes=["status-card"],
                    )
                    buffer_display = gr.Textbox(
                        label="🔄 Buffer",
                        value="0.00 s",
                        interactive=False,
                        elem_classes=["status-card"],
                    )

                speech_indicator = gr.Textbox(
                    label="🗣️ Voice Activity",
                    value="⬤ Waiting for input...",
                    interactive=False,
                    elem_classes=["status-card"],
                )

            # ── Right Column: Emotion Display & Timeline ─────
            with gr.Column(scale=2):
                # Current Emotion
                emotion_html = gr.HTML(
                    value="""
                    <div class="emotion-display">
                        <div style="font-size:4em;">😐</div>
                        <div style="font-size:1.8em; font-weight:600; color:#808080; margin-top:10px;">
                            Neutral
                        </div>
                        <div style="font-size:0.9em; color:#666; margin-top:5px;">
                            Confidence: 0.0%
                        </div>
                    </div>
                    """,
                )

                # Timeline Chart
                timeline_chart = gr.Plot(
                    value=_create_empty_timeline_chart(),
                    label="Emotion Timeline",
                )

                # Emotion History Log
                emotion_log = gr.Dataframe(
                    headers=["Time", "Emotion", "Confidence", "Arousal", "Dominance", "Valence"],
                    datatype=["str", "str", "number", "number", "number", "number"],
                    label="📋 Session Log (Last 20)",
                    row_count=5,
                    interactive=False,
                )

        # ── Streaming Callback ───────────────────────────────
        def process_audio_stream(audio_data, history_state):
            """Process streaming audio from microphone."""
            if audio_data is None:
                return [
                    gr.update(),   # emotion_html
                    gr.update(),   # timeline_chart
                    gr.update(),   # dimension_chart
                    gr.update(),   # latency_display
                    gr.update(),   # buffer_display
                    gr.update(),   # speech_indicator
                    gr.update(),   # emotion_log
                    history_state,
                ]

            sr, audio_array = audio_data

            # Convert to float32 mono
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # Resample if needed (Gradio typically sends at 48kHz)
            if sr != processor.sample_rate:
                # Simple decimation for speed
                ratio = sr / processor.sample_rate
                indices = np.arange(0, len(audio_array), ratio).astype(int)
                indices = indices[indices < len(audio_array)]
                audio_array = audio_array[indices]

            # Process through pipeline
            result = processor.process_chunk(audio_array)

            # Update timeline
            elapsed = time.time() - start_time[0]
            timeline_times.append(round(elapsed, 1))
            timeline_arousal.append(result["arousal"])
            timeline_dominance.append(result["dominance"])
            timeline_valence.append(result["valence"])
            timeline_labels.append(f'{result["emoji"]} {result["emotion_label"]}')

            # Build timeline chart
            fig_timeline = go.Figure()
            times_list = list(timeline_times)

            for data, name, color in [
                (list(timeline_arousal), "Arousal", "#FF6B6B"),
                (list(timeline_dominance), "Dominance", "#4ECDC4"),
                (list(timeline_valence), "Valence", "#45B7D1"),
            ]:
                fig_timeline.add_trace(go.Scatter(
                    x=times_list, y=data,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)",
                ))

            fig_timeline.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,30,0.8)",
                margin=dict(l=40, r=20, t=40, b=40),
                height=300,
                title=dict(
                    text="Emotion Dimensions Timeline",
                    font=dict(size=14, color="#aaa"),
                ),
                xaxis=dict(
                    title="Time (s)",
                    gridcolor="rgba(255,255,255,0.1)",
                    zeroline=False,
                ),
                yaxis=dict(
                    title="Value",
                    range=[0, 1],
                    gridcolor="rgba(255,255,255,0.1)",
                    zeroline=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11),
                ),
            )

            # Emotion display HTML
            color = result["color"]
            confidence_pct = result["confidence"] * 100
            emotion_html_val = f"""
            <div class="emotion-display" style="border-color: {color}40;">
                <div style="font-size:4.5em; filter: drop-shadow(0 0 20px {color}80);">
                    {result['emoji']}
                </div>
                <div style="font-size:2em; font-weight:700; color:{color};
                            margin-top:10px; text-transform:uppercase; letter-spacing:2px;">
                    {result['emotion_label']}
                </div>
                <div style="margin-top:12px;">
                    <div style="background:rgba(255,255,255,0.1); border-radius:10px;
                                height:8px; width:80%; margin:0 auto; overflow:hidden;">
                        <div style="background:linear-gradient(90deg, {color}88, {color});
                                    height:100%; width:{confidence_pct}%;
                                    border-radius:10px; transition:width 0.3s;"></div>
                    </div>
                    <div style="font-size:0.85em; color:#888; margin-top:6px;">
                        Confidence: {confidence_pct:.1f}%
                    </div>
                </div>
            </div>
            """

            # Dimension bar chart
            fig_bars = _create_emotion_bar_chart(result)

            # Latency
            latency_str = f"{result['latency_ms']:.0f} ms"

            # Buffer
            buffer_str = f"{result['buffer_seconds']:.2f} s"

            # Speech indicator
            if result["is_speech"]:
                speech_str = "🟢 Speech Detected"
            else:
                speech_str = "🔴 Silence / Noise"

            # Update history
            if history_state is None:
                history_state = []

            if result["is_speech"]:
                history_state.insert(0, [
                    f"{elapsed:.1f}s",
                    f"{result['emoji']} {result['emotion_label']}",
                    round(result["confidence"], 3),
                    round(result["arousal"], 3),
                    round(result["dominance"], 3),
                    round(result["valence"], 3),
                ])
                history_state = history_state[:20]  # Keep last 20

            return [
                emotion_html_val,     # emotion_html
                fig_timeline,         # timeline_chart
                fig_bars,             # dimension_chart
                latency_str,          # latency_display
                buffer_str,           # buffer_display
                speech_str,           # speech_indicator
                history_state if history_state else gr.update(),
                history_state,
            ]

        # State for history
        history_state = gr.State([])

        # Wire up streaming
        audio_input.stream(
            fn=process_audio_stream,
            inputs=[audio_input, history_state],
            outputs=[
                emotion_html,
                timeline_chart,
                dimension_chart,
                latency_display,
                buffer_display,
                speech_indicator,
                emotion_log,
                history_state,
            ],
            stream_every=0.5,
            time_limit=300,
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:20px; color:#555; font-size:0.8em; border-top:1px solid rgba(255,255,255,0.05); margin-top:30px;">
            VoxDynamics v1.0 • Powered by Wav2Vec2 + Silero VAD • Language-Agnostic Emotion Recognition
        </div>
        """)

    return demo
