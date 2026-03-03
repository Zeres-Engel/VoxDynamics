# ============================================================
# VoxDynamics — Analytics View Component
# ============================================================

import httpx
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px

API_BASE_URL = "http://localhost:8000/api"


def create_analytics_view() -> dict:
    """Build the Session Analytics View. Returns dict with outputs and load_fn."""
    
    gr.HTML("""
        <div style="text-align:center; margin:15px 0 10px 0;">
            <span style="font-size:1.2em; font-weight:600; color:#00e5ff; letter-spacing:2px;">📊 SESSION ANALYTICS</span>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.HTML('<div style="color:#00e5ff; font-weight:600; margin-bottom:10px; letter-spacing:1px; font-size:0.85em;">📈 SUMMARY</div>')
            
            avg_arousal = gr.Textbox(label="Avg Arousal", value="--", interactive=False, elem_classes=["status-card"])
            avg_dominance = gr.Textbox(label="Avg Dominance", value="--", interactive=False, elem_classes=["status-card"])
            avg_valence = gr.Textbox(label="Avg Valence", value="--", interactive=False, elem_classes=["status-card"])
            total_duration = gr.Textbox(label="Data Points", value="--", interactive=False, elem_classes=["status-card"])
            
        with gr.Column(scale=2, elem_classes=["glass-panel"]):
            gr.HTML('<div style="color:#00e5ff; font-weight:600; margin-bottom:10px; letter-spacing:1px; font-size:0.85em;">🥧 EMOTION DISTRIBUTION</div>')
            pie_chart = gr.Plot(label=None)
            
            gr.HTML('<div style="color:#00e5ff; font-weight:600; margin-top:15px; margin-bottom:10px; letter-spacing:1px; font-size:0.85em;">🌊 FULL LIFECYCLE TIMELINE</div>')
            history_timeline = gr.Plot(label=None)

    def load_session_data(session_uuid):
        if not session_uuid:
            return [gr.update(value="--")] * 4 + [gr.update(), gr.update()]
            
        try:
            response = httpx.get(f"{API_BASE_URL}/emotions/{session_uuid}?limit=1000")
            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    return [gr.update(value="0")] * 4 + [gr.update(), gr.update()]
                    
                df = pd.DataFrame(data)
                
                pts = len(df)
                a_mean = df['arousal'].mean()
                d_mean = df['dominance'].mean()
                v_mean = df['valence'].mean()
                
                # Pie Chart
                emotion_counts = df['emotion_label'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']
                
                fig_pie = px.pie(
                    emotion_counts, 
                    values='Count', 
                    names='Emotion',
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Tealgrn
                )
                fig_pie.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=250
                )
                
                # Timeline
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                start_t = df['timestamp'].min()
                df['seconds'] = (df['timestamp'] - start_t).dt.total_seconds()
                
                fig_timeline = go.Figure()
                for dim, color in [
                    ("arousal", "#00d2ff"),
                    ("dominance", "#3a7bd5"),
                    ("valence", "#00ff87"),
                ]:
                    fig_timeline.add_trace(go.Scatter(
                        x=df['seconds'], y=df[dim],
                        mode="lines",
                        name=dim.capitalize(),
                        line=dict(color=color, width=2, shape="spline"),
                        fill="tozeroy",
                        fillcolor=f"rgba({int(color[1:3],16) if color.startswith('#') and len(color)==7 else 0},0,0,0.05)",
                    ))
                
                fig_timeline.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.02)",
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=280,
                    xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Value", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
                )
                
                return [
                    gr.update(value=f"{a_mean:.3f}"),
                    gr.update(value=f"{d_mean:.3f}"),
                    gr.update(value=f"{v_mean:.3f}"),
                    gr.update(value=str(pts)),
                    fig_pie,
                    fig_timeline
                ]
        except Exception as e:
            print(f"Error processing session data: {e}")
            
        return [gr.update(value="Error")] * 4 + [gr.update(), gr.update()]

    return {
        "outputs": [avg_arousal, avg_dominance, avg_valence, total_duration, pie_chart, history_timeline],
        "load_fn": load_session_data
    }
