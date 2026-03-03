# ============================================================
# VoxDynamics — Main Gradio Dashboard (Google Meet Layout)
# ============================================================
"""
Vertical scroll layout:
  Section 1 — Hero Stage (Live Recording, centered)
  Section 2 — Archive (Session History + Analytics)
"""

import httpx
import gradio as gr
from app.core.processor import AudioProcessor
from app.ui.components import CUSTOM_CSS
from app.ui.tabs.realtime_tab import create_realtime_view
from app.ui.tabs.analytics_tab import create_analytics_view

API_BASE_URL = "http://localhost:8000/api"


def fetch_sessions_for_table():
    """Fetch structured session data from API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/sessions")
        if response.status_code == 200:
            data = response.json().get("sessions", [])
            if not data:
                return []
            rows = []
            for item in data:
                rows.append([
                    item["UUID"], item["Time"], item["Dur."],
                    str(item["A"]), str(item["D"]), str(item["V"])
                ])
            return rows
    except Exception as e:
        print(f"Error fetching sessions: {e}")
    return []


def create_dashboard(processor: AudioProcessor) -> gr.Blocks:
    """Build the main Gradio Blocks dashboard — Google Meet style."""

    with gr.Blocks(
        title="VoxDynamics — Real-Time SER",
        theme=gr.themes.Base(
            primary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Outfit"),
        ),
        css=CUSTOM_CSS,
    ) as demo:

        # ══════════════════════════════════════════════════
        # SECTION 1 — THE STAGE (Live Recording)
        # ══════════════════════════════════════════════════
        create_realtime_view(processor)

        # ══════════════════════════════════════════════════
        # DIVIDER
        # ══════════════════════════════════════════════════
        gr.HTML("""
            <div class="section-divider">
                ─── scroll down for session history ───
            </div>
        """)

        # ══════════════════════════════════════════════════
        # SECTION 2 — THE ARCHIVE (Session History)
        # ══════════════════════════════════════════════════
        selected_session_uuid = gr.State("")

        with gr.Column(elem_classes=["archive-section"]):
            gr.HTML("""
                <div style="text-align:center; margin-bottom:15px;">
                    <span style="font-size:1.6em; font-weight:700; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📚 SESSION ARCHIVE</span>
                </div>
            """)

            refresh_btn = gr.Button("🔄 Refresh Sessions", elem_classes=["btn-primary"])

            sessions_df = gr.Dataframe(
                headers=["UUID", "Time", "Dur.", "Avg A", "Avg D", "Avg V"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            # ── Analytics Panel (hidden until a session is selected) ──
            with gr.Column(visible=False) as analytics_container:
                analytics_refs = create_analytics_view()

        # ── Events ───────────────────────────────────────

        # Load sessions on app start
        demo.load(
            fn=fetch_sessions_for_table,
            inputs=[],
            outputs=[sessions_df]
        )

        # Refresh button
        refresh_btn.click(
            fn=fetch_sessions_for_table,
            inputs=[],
            outputs=[sessions_df]
        )

        # Click on a session row → show analytics
        def on_row_select(evt: gr.SelectData, current_data):
            row_idx = evt.index[0]
            try:
                clicked_uuid = current_data.iloc[row_idx, 0]
            except:
                try:
                    clicked_uuid = current_data[row_idx][0]
                except:
                    clicked_uuid = ""

            if not clicked_uuid:
                return gr.update(), gr.update(visible=False)

            return (
                clicked_uuid,
                gr.update(visible=True),
            )

        sessions_df.select(
            fn=on_row_select,
            inputs=[sessions_df],
            outputs=[selected_session_uuid, analytics_container]
        ).then(
            fn=analytics_refs["load_fn"],
            inputs=[selected_session_uuid],
            outputs=analytics_refs["outputs"]
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:25px; color:#333; font-size:0.8em; font-weight:500; letter-spacing:2px; margin-top:15px;">
            VOXDYNAMICS • SPEECH EMOTION INTELLIGENCE PLATFORM
        </div>
        """)

    return demo
