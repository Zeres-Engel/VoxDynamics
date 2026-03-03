# ============================================================
# VoxDynamics — Real-Time Streaming View (Google Meet Style)
# ============================================================

import time
import uuid
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from collections import deque

from app.core.processor import AudioProcessor
from app.db.database import start_session, end_session
from app.ui.components import create_empty_timeline_chart, create_emotion_bar_chart, MAX_TIMELINE_POINTS


def create_realtime_view(processor: AudioProcessor):
    """Build the center-stage recording view inspired by Google Meet."""
    
    # State containers
    timeline_arousal = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_dominance = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_valence = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_times = deque(maxlen=MAX_TIMELINE_POINTS)
    timeline_labels = deque(maxlen=MAX_TIMELINE_POINTS)
    start_time = [time.time()]

    # ── Header ───────────────────────────────────────────
    gr.HTML("""
        <div class="main-title">VOXDYNAMICS</div>
        <div class="subtitle">Real-Time Speech Emotion Recognition</div>
    """)
    
    # State variables for session management
    current_session_uuid = gr.State("")
    is_session_active = gr.State(False)

    # ══════════════════════════════════════════════════════
    # HERO STAGE — Center of the screen
    # ══════════════════════════════════════════════════════
    with gr.Column(elem_classes=["hero-stage"]):
        
        with gr.Row():
            # ── Left: Live Dimension Gauges ──────────────
            with gr.Column(scale=1):
                dimension_chart = gr.Plot(
                    label=None,
                    value=create_emotion_bar_chart(
                        {"arousal": 0.0, "dominance": 0.0, "valence": 0.0}
                    ),
                )
                with gr.Row():
                    speech_indicator = gr.Textbox(
                        label="Voice",
                        value="⬤ Waiting...",
                        interactive=False,
                        elem_classes=["status-card"],
                    )
                    latency_display = gr.Textbox(
                        label="Latency",
                        value="-- ms",
                        interactive=False,
                        elem_classes=["status-card"],
                    )
            
            # ── Center: Emotion Display (The Star) ───────
            with gr.Column(scale=2):
                emotion_html = gr.HTML(
                    value="""
                    <div class="emotion-display" style="border: 2px dashed rgba(255,255,255,0.15);">
                        <div style="font-size:6em; opacity: 0.4;">🎙️</div>
                        <div style="font-size:2.2em; font-weight:700; color:#444; margin-top:10px;">
                            AWAITING SIGNAL
                        </div>
                        <div style="font-size:0.95em; color:#666; margin-top:8px;">
                            Press Record to begin a session
                        </div>
                    </div>
                    """,
                )

            # ── Right: Live Log Stream ───────────────────
            with gr.Column(scale=1):
                gr.HTML('<div style="color:#00e5ff; font-weight:600; letter-spacing:1px; text-align:center; margin-bottom:10px; font-size:0.85em;">📃 LIVE LOG</div>')
                emotion_log = gr.Dataframe(
                    headers=["T", "Emotion", "Conf", "A", "D", "V"],
                    datatype=["str", "str", "number", "number", "number", "number"],
                    label=None,
                    row_count=5,
                    interactive=False,
                    elem_id="emotion-log",
                )

        # ── Microphone Input — Centered Below Hero ───────
        gr.HTML('<div style="text-align:center; margin-top:20px; margin-bottom:5px; color:#00e5ff; font-weight:600; letter-spacing:2px; font-size:0.9em;">🎤 SIGNAL SOURCE</div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label=None,
                    elem_id="mic-input",
                    interactive=True,
                )
                session_status = gr.Markdown("<div style='text-align:center; color:#555; margin-top:8px; font-size:0.9em;'>📡 Session Offline</div>")
            with gr.Column(scale=1):
                pass

    # ══════════════════════════════════════════════════════
    # TIMELINE — Full-width chart below the stage
    # ══════════════════════════════════════════════════════
    with gr.Column(elem_classes=["glass-panel"]):
        gr.HTML('<div style="color:#00e5ff; font-weight:600; letter-spacing:1px; text-align:center; margin-bottom:5px; font-size:0.85em;">🌊 EMOTION TIMELINE</div>')
        timeline_chart = gr.Plot(
            label=None,
            value=create_empty_timeline_chart(),
            elem_id="dimension-chart"
        )

    # ── Auto-Session Callbacks ───────────────────────────
    
    async def on_start_recording():
        """Triggered when the user clicks 'Record'."""
        session_id = str(uuid.uuid4())
        try:
            # CRITICAL: Reset processor state so old EMA data doesn't bleed in
            processor.reset()
            processor.session_id = session_id
            
            await start_session(session_id)
            
            timeline_arousal.clear()
            timeline_dominance.clear()
            timeline_valence.clear()
            timeline_times.clear()
            timeline_labels.clear()
            start_time[0] = time.time()
            
            return (
                f"<div style='text-align:center; color:#00ff87; font-weight:600; font-size:0.9em;'>🟢 Recording — Session: {session_id[:8]}...</div>",
                session_id,
                True
            )
        except Exception as e:
            print(f"Session start error: {e}")
            return f"<div style='text-align:center; color:#ff416c; font-size:0.9em;'>🔴 Error: {e}</div>", "", False

    audio_input.start_recording(
        fn=on_start_recording,
        inputs=[],
        outputs=[session_status, current_session_uuid, is_session_active]
    )
    
    async def on_stop_recording(sess_uuid):
        """Triggered when the user clicks 'Stop' or 'Clear'."""
        if not sess_uuid:
            return gr.update(), gr.update(), gr.update()
            
        try:
            await end_session(sess_uuid)
            return (
                f"<div style='text-align:center; color:#888; font-size:0.9em;'>🔴 Session Saved: {sess_uuid[:8]}...</div>",
                "",
                False
            )
        except Exception as e:
            print(f"Session stop error: {e}")
            return gr.update(), sess_uuid, True

    audio_input.stop_recording(
        fn=on_stop_recording,
        inputs=[current_session_uuid],
        outputs=[session_status, current_session_uuid, is_session_active]
    )
    audio_input.clear(
        fn=on_stop_recording,
        inputs=[current_session_uuid],
        outputs=[session_status, current_session_uuid, is_session_active]
    )

    # ── Streaming Callback ───────────────────────────────
    def process_audio_stream(audio_data, history_state, is_active):
        """Process streaming audio from microphone."""
        try:
            if not is_active:
                return [gr.update()] * 7 + [history_state]
                
            if history_state is None:
                history_state = []

            if audio_data is None:
                return [
                    gr.update(),   # emotion_html
                    gr.update(),   # timeline_chart
                    gr.update(),   # speech_indicator
                    gr.update(),   # latency_display
                    gr.update(),   # dimension_chart
                    gr.update(),   # emotion_log
                    gr.update(),   # session_status placeholder
                    history_state,
                ]

            sr, audio_array = audio_data

            # Convert to float32 mono
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            if sr != processor.sample_rate:
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

            fig_timeline = go.Figure()
            times_list = list(timeline_times)

            for data, name, color in [
                (list(timeline_arousal), "Arousal", "#f9a825"),
                (list(timeline_dominance), "Dominance", "#ab47bc"),
                (list(timeline_valence), "Valence", "#00e5ff"),
            ]:
                r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig_timeline.add_trace(go.Scatter(
                    x=times_list, y=data,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2.5, shape="spline"),
                    fill="tozeroy",
                    fillcolor=f"rgba({r},{g},{b},0.07)",
                ))

            fig_timeline.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10,10,25,0.6)",
                margin=dict(l=40, r=20, t=10, b=40),
                height=220,
                xaxis=dict(
                    title=dict(text="Time (s)", font=dict(size=11, color="#556")),
                    gridcolor="rgba(255,255,255,0.04)",
                    tickfont=dict(color="#556"),
                ),
                yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#556")),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right", font=dict(size=11, color="#aaa"), bgcolor="rgba(0,0,0,0)"),
            )

            color = result["color"]
            confidence_pct = result["confidence"] * 100
            emotion_html_val = f"""
            <div class="emotion-display" style="border: 1px solid {color}40; box-shadow: 0 0 50px {color}10;">
                <div style="font-size:6em; filter: drop-shadow(0 0 30px {color}aa);">
                    {result['emoji']}
                </div>
                <div style="font-size:2.5em; font-weight:800; color:{color};
                            margin-top:10px; text-transform:uppercase; letter-spacing:4px;
                            text-shadow: 0 0 15px {color}30;">
                    {result['emotion_label']}
                </div>
                <div style="margin-top:18px;">
                    <div style="background:rgba(255,255,255,0.04); border-radius:12px;
                                height:8px; width:60%; margin:0 auto; overflow:hidden; border: 1px solid rgba(255,255,255,0.08);">
                        <div style="background:linear-gradient(90deg, {color}44, {color});
                                    height:100%; width:{confidence_pct}%;
                                    border-radius:12px; transition:width 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);"></div>
                    </div>
                    <div style="font-size:0.9em; color:#aaa; margin-top:8px; font-weight:500;">
                        CONFIDENCE: {confidence_pct:.1f}%
                    </div>
                </div>
            </div>
            """

            fig_bars = create_emotion_bar_chart(result)
            latency_str = f"{result['latency_ms']:.0f} ms"
            speech_str = "🟢 ACTIVE" if result["is_speech"] else "🔘 IDLE"

            if result["is_speech"]:
                history_state.insert(0, [
                    f"{elapsed:.1f}s",
                    f"{result['emoji']} {result['emotion_label']}",
                    round(result["confidence"], 2),
                    round(result["arousal"], 2),
                    round(result["dominance"], 2),
                    round(result["valence"], 2),
                ])
                history_state = history_state[:15]

            return [
                emotion_html_val,
                fig_timeline,
                speech_str,
                latency_str,
                fig_bars,
                history_state if history_state else gr.update(),
                gr.update(),
                history_state,
            ]
        except Exception as e:
            import traceback
            print(f"❌ Error in process_audio_stream: {e}")
            traceback.print_exc()
            return [gr.update()] * 7 + [history_state]

    # State for history
    history_state = gr.State([])

    # Wire up streaming
    audio_input.stream(
        fn=process_audio_stream,
        inputs=[audio_input, history_state, is_session_active],
        outputs=[
            emotion_html,
            timeline_chart,
            speech_indicator,
            latency_display,
            dimension_chart,
            emotion_log,
            session_status,
            history_state,
        ],
        stream_every=0.3,
        time_limit=600,
    )
