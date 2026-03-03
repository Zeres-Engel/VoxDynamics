# ============================================================
# VoxDynamics — UI Components & Styles
# ============================================================
"""Shared CSS and plotting utilities for the Gradio interface."""

import plotly.graph_objects as go

# ── Custom CSS ───────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ────────────────────────────────────── */
.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    max-width: 100% !important;
    background: linear-gradient(180deg, #0a0a1a 0%, #111128 40%, #0a0a1a 100%) !important;
    min-height: 100vh;
}

/* ── Hero Stage — The Google Meet Area ───────────────── */
.hero-stage {
    background: radial-gradient(ellipse at 50% 0%, rgba(0, 180, 255, 0.06) 0%, rgba(100, 50, 255, 0.04) 40%, transparent 80%) !important;
    border: 1px solid rgba(0, 200, 255, 0.1) !important;
    border-radius: 28px !important;
    padding: 30px 20px !important;
    margin-bottom: 10px !important;
    position: relative !important;
}

.hero-stage::before {
    content: '' !important;
    position: absolute !important;
    inset: -1px !important;
    border-radius: 28px !important;
    padding: 1px !important;
    background: linear-gradient(135deg, rgba(0,210,255,0.2), transparent 40%, rgba(171,71,188,0.15));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
    -webkit-mask-composite: xor !important;
    pointer-events: none !important;
}

.main-title {
    text-align: center;
    padding-top: 15px;
    background: linear-gradient(90deg, #60efff, #00ff87, #0061ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: rgba(102, 126, 234, 0.8);
    font-size: 0.95em;
    font-weight: 300;
    margin-bottom: 25px;
    text-transform: uppercase;
    letter-spacing: 4px;
}

/* ── Emotion Display — Center Stage Card ─────────────── */
.emotion-display {
    text-align: center;
    padding: 35px 20px;
    border-radius: 24px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.4s ease;
}

/* ── Glass Panel — General Cards ─────────────────────── */
.glass-panel {
    background: rgba(255, 255, 255, 0.025) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px 0 rgba(0, 0, 0, 0.5) !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

.glass-panel:hover {
    border-color: rgba(255, 255, 255, 0.14) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

/* ── Status Cards ────────────────────────────────────── */
.status-card {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,210,255,0.3) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

.btn-danger {
    background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── Archive Section — Below the Fold ────────────────── */
.archive-section {
    margin-top: 20px !important;
    padding: 25px !important;
    background: rgba(255,255,255,0.015) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 24px !important;
}

.section-divider {
    text-align: center;
    padding: 30px 0 15px 0;
    color: rgba(255,255,255,0.3);
    font-size: 0.85em;
    letter-spacing: 6px;
    text-transform: uppercase;
    position: relative;
}

.section-divider::before,
.section-divider::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 30%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.section-divider::before { left: 5%; }
.section-divider::after { right: 5%; }

/* ── Mic Input Styling ───────────────────────────────── */
#mic-input {
    background: transparent !important;
    max-width: 500px;
    margin: 0 auto;
}
#emotion-log, #dimension-chart {
    background: transparent !important;
}

.gr-box { border-radius: 15px !important; }

/* ── Live Indicator Badge ────────────────────────────── */
.live-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 600;
    letter-spacing: 1px;
}
"""

# ── Timeline buffer for charts ──────────────────────────────
MAX_TIMELINE_POINTS = 150

def create_empty_timeline_chart() -> go.Figure:
    """Create an empty Plotly timeline chart."""
    fig = go.Figure()

    for dim, color in [
        ("Arousal", "#f9a825"),     # Warm gold
        ("Dominance", "#ab47bc"),   # Violet
        ("Valence", "#00e5ff"),     # Bright cyan
    ]:
        fig.add_trace(go.Scatter(
            x=[], y=[], mode="lines",
            name=dim, line=dict(color=color, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,25,0.6)",
        margin=dict(l=40, r=20, t=10, b=40),
        height=220,
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=11, color="#556")),
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            tickfont=dict(color="#556"),
        ),
        yaxis=dict(
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            tickfont=dict(color="#556"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#aaa"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )
    return fig

def create_emotion_bar_chart(result: dict) -> go.Figure:
    """Create a horizontal bar chart showing emotion dimensions."""
    dims = ["Arousal", "Dominance", "Valence"]
    vals = [result.get("arousal", 0), result.get("dominance", 0), result.get("valence", 0)]
    colors = ["#f9a825", "#ab47bc", "#00e5ff"]   # gold, violet, cyan

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=dims, x=vals,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.15)", width=0),
        ),
        text=[f"{v:.2f}" for v in vals],
        textposition="inside",
        textfont=dict(size=13, color="white", family="Outfit"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,25,0.6)",
        margin=dict(l=78, r=10, t=8, b=8),
        height=130,
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(tickfont=dict(size=13, color="#ccc", family="Outfit")),
        showlegend=False,
    )
    return fig
