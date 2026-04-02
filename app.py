"""
app.py - FinLearn AI Tutor
Run: streamlit run app.py
"""

from time import sleep

import plotly.graph_objects as go
import streamlit as st

from inference import run_simulation


st.set_page_config(
    page_title="AI Trading Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0e1117;
    --panel: rgba(22, 27, 37, 0.78);
    --panel-strong: rgba(18, 22, 31, 0.92);
    --card: rgba(255, 255, 255, 0.05);
    --card-soft: rgba(255, 255, 255, 0.03);
    --border: rgba(255, 255, 255, 0.09);
    --text: #f3f6fb;
    --muted: #98a2b3;
    --green: #37d39a;
    --red: #ff6b81;
    --blue: #5aa7ff;
    --blue-soft: rgba(90, 167, 255, 0.14);
    --amber: #ffbf69;
    --amber-soft: rgba(255, 191, 105, 0.14);
    --purple: #a78bfa;
    --purple-soft: rgba(167, 139, 250, 0.14);
    --shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(circle at top left, rgba(90, 167, 255, 0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.12), transparent 18%),
        linear-gradient(180deg, #0e1117 0%, #0b0e14 100%);
    color: var(--text);
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    max-width: 1440px;
    padding: 1.3rem 1.5rem 2rem;
}

[data-testid="stSidebar"] {
    background: #0d121a !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] * {
    color: #e8edf7 !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stButton"] > button {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: linear-gradient(135deg, rgba(90, 167, 255, 0.95), rgba(55, 211, 154, 0.95)) !important;
    color: #071019 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 0 16px 34px rgba(51, 153, 255, 0.24) !important;
}

[data-testid="stSidebar"] .stToggle label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectSlider label {
    font-weight: 600 !important;
}

.hero-shell {
    background: linear-gradient(135deg, rgba(17, 23, 33, 0.92), rgba(13, 18, 26, 0.94));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 1.35rem 1.45rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}

.hero-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}

.hero-kicker {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.68rem;
    color: #8db9ff;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.35rem;
    line-height: 1;
    letter-spacing: -0.03em;
    margin: 0.4rem 0 0.5rem;
    color: var(--text);
}

.hero-title span {
    color: var(--green);
}

.hero-copy {
    max-width: 760px;
    color: #acb7c8;
    font-size: 0.95rem;
    line-height: 1.65;
}

.hero-badges {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
}

.badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.09);
    color: #d8e0ee;
    border-radius: 999px;
    padding: 0.3rem 0.7rem;
    font-size: 0.72rem;
}

.glass-card {
    background: var(--panel);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1rem;
    box-shadow: var(--shadow);
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.08rem;
    color: var(--text);
    margin-bottom: 0.18rem;
}

.section-subtitle {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.5;
    margin-bottom: 0.9rem;
}

.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.035));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    min-height: 120px;
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.62rem;
    color: #90a0b4;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    line-height: 1;
    margin-top: 0.55rem;
    color: var(--text);
}

.metric-sub {
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.45rem;
}

.thinking-box,
.concept-box,
.suggestion-box,
.blackbox-box {
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-top: 0.7rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.thinking-box {
    background: linear-gradient(180deg, rgba(90, 167, 255, 0.14), rgba(90, 167, 255, 0.08));
}

.concept-box {
    background: linear-gradient(180deg, rgba(167, 139, 250, 0.14), rgba(167, 139, 250, 0.08));
}

.suggestion-box {
    background: linear-gradient(180deg, rgba(55, 211, 154, 0.14), rgba(55, 211, 154, 0.08));
}

.blackbox-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.04));
}

.box-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.62rem;
    color: #bed4f8;
    margin-bottom: 0.45rem;
}

.concept-box .box-label {
    color: #d5c7ff;
}

.suggestion-box .box-label {
    color: #9ef0cd;
}

.blackbox-box .box-label {
    color: #a9b5c6;
}

.box-copy {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #eef3fb;
}

.status-strip {
    margin-top: 0.9rem;
    padding: 0.7rem 0.85rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
}

.status-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #93a6bf;
    text-transform: uppercase;
}

.status-value {
    font-size: 0.92rem;
    margin-top: 0.3rem;
    color: #edf3fd;
}

.feed-wrap {
    max-height: 760px;
    overflow-y: auto;
    padding-right: 0.25rem;
}

.feed-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.035));
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
}

.feed-top {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.feed-step,
.feed-action {
    border-radius: 999px;
    padding: 0.28rem 0.62rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
}

.feed-step {
    background: rgba(255,255,255,0.06);
    color: #b3c0d2;
}

.feed-action {
    border: 1px solid rgba(255,255,255,0.08);
}

.action-buy {
    color: #95f1ca;
    background: rgba(55, 211, 154, 0.12);
}

.action-sell {
    color: #ffadbb;
    background: rgba(255, 107, 129, 0.13);
}

.action-hold {
    color: #d9e1ec;
    background: rgba(255,255,255,0.06);
}

.action-rebalance {
    color: #ffd28d;
    background: rgba(255, 191, 105, 0.13);
}

.feed-value {
    margin-left: auto;
    color: #f3f7ff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}

.feed-meta {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0.55rem;
}

.empty-state {
    padding: 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.08);
    color: #a9b5c6;
    font-size: 0.9rem;
}

.footer-note {
    text-align: center;
    color: #6f7c8f;
    font-size: 0.72rem;
    margin-top: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em;
}
</style>
""",
    unsafe_allow_html=True,
)


def clean(text: str) -> str:
    for prefix in (
        "📘 Concept:",
        "📘 ",
        "📌 Reasoning:",
        "📌 ",
        "➡ Suggestion:",
        "➡  Suggestion:",
        "➡ ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def action_class(action: str) -> str:
    upper = action.upper()
    if "BUY" in upper:
        return "action-buy"
    if "SELL" in upper:
        return "action-sell"
    if "REBALANCE" in upper:
        return "action-rebalance"
    return "action-hold"


def metric_card(label: str, value: str, sub: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def build_portfolio_chart(steps, initial_value, current_index):
    shown_steps = steps[:current_index]
    xs = [0] + [item["step"] for item in shown_steps]
    ys = [initial_value] + [item["portfolio_value"] for item in shown_steps]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#67b4ff", width=4, shape="spline", smoothing=1.0),
            fill="tozeroy",
            fillcolor="rgba(103, 180, 255, 0.16)",
            hovertemplate="Step %{x}<br>Portfolio $%{y:,.2f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_hline(
        y=initial_value,
        line_dash="dot",
        line_color="rgba(255, 191, 105, 0.65)",
        line_width=1.4,
    )

    if shown_steps:
        latest = shown_steps[-1]
        fig.add_trace(
            go.Scatter(
                x=[latest["step"]],
                y=[latest["portfolio_value"]],
                mode="markers",
                marker=dict(size=12, color="#37d39a", line=dict(color="#0e1117", width=2)),
                hovertemplate="Latest move<br>Step %{x}<br>$%{y:,.2f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=480,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#93a6bf"),
        xaxis=dict(
            title="Step",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Portfolio Value",
            tickprefix="$",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        hovermode="x unified",
    )
    return fig


def render_ai_panel(step, explainable_mode, total_steps):
    st.markdown('<div class="section-title">AI Decision Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">The left panel is the mentor brain. It surfaces the latest decision and teaches the user what the AI is seeing.</div>',
        unsafe_allow_html=True,
    )

    if step is None:
        st.markdown(
            """
            <div class="empty-state">
                Run the simulation to activate the AI Decision Engine. The latest reasoning, concept, and suggestion will appear here in real time.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    action = step["action"].replace("_", " ")
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="status-label">Latest Decision</div>
            <div class="status-value" style="font-family:'Space Grotesk',sans-serif;font-size:1.3rem;color:#f3f6fb;margin-top:0.45rem">
                Step {step['step']} of {total_steps} • {action}
            </div>
            <div class="status-strip">
                <div class="status-label">Portfolio Snapshot</div>
                <div class="status-value">${step['portfolio_value']:,.2f}</div>
            </div>
            {
                f'''
                <div class="thinking-box">
                    <div class="box-label">AI Reasoning</div>
                    <div class="box-copy">{clean(step["reasoning"])}</div>
                </div>
                <div class="concept-box">
                    <div class="box-label">Concept</div>
                    <div class="box-copy">{clean(step["concept"])}</div>
                </div>
                <div class="suggestion-box">
                    <div class="box-label">Suggestion</div>
                    <div class="box-copy">{clean(step["suggestion"])}</div>
                </div>
                '''
                if explainable_mode
                else
                '''
                <div class="blackbox-box">
                    <div class="box-label">Explainable AI Mode Off</div>
                    <div class="box-copy">The system still acts, but the reasoning is intentionally hidden. This makes the product feel opaque instead of trustworthy.</div>
                </div>
                '''
            }
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feed(steps, current_index):
    st.markdown('<div class="section-title">Decision Feed</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A live log of every move the system has made so far.</div>',
        unsafe_allow_html=True,
    )

    revealed = list(reversed(steps[:current_index]))
    if not revealed:
        st.markdown(
            """
            <div class="empty-state">
                The command feed will populate after the first simulation step.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cards = ['<div class="feed-wrap">']
    for step in revealed:
        action = step["action"].replace("_", " ")
        cards.append(
            f"""
            <div class="feed-card">
                <div class="feed-top">
                    <span class="feed-step">STEP {step['step']:02d}</span>
                    <span class="feed-action {action_class(step['action'])}">{action}</span>
                    <span class="feed-value">${step['portfolio_value']:,.2f}</span>
                </div>
                <div class="feed-meta">Reward {step['reward']:+.4f}</div>
            </div>
            """
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0
if "playing" not in st.session_state:
    st.session_state.playing = False


with st.sidebar:
    st.markdown("## Command Controls")
    max_steps = st.slider("Simulation Steps", 10, 40, 20, 5)
    seed = st.slider("Market Seed", 1, 99, 42, 1)
    explainable_mode = st.toggle("Explainable AI Mode", value=True)
    playback_speed = st.select_slider("Playback Speed", options=["Slow", "Normal", "Fast"], value="Normal")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.simulation_data = None
        st.session_state.current_step = 0
        st.session_state.playing = False
        st.rerun()


speed_map = {"Slow": 0.85, "Normal": 0.45, "Fast": 0.18}

top_left, top_right = st.columns([3.5, 1], gap="large")
with top_left:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-row">
                <div>
                    <div class="hero-kicker">Meta Hackathon • AI Trading Command Center</div>
                    <div class="hero-title">FinLearn <span>Command Center</span></div>
                    <div class="hero-copy">
                        A premium fintech-style simulation where the AI makes portfolio decisions in real time,
                        visualizes capital movement, and optionally reveals the logic behind every action.
                    </div>
                    <div class="hero-badges">
                        <span class="badge">Live Portfolio Tracking</span>
                        <span class="badge">AI Decision Feed</span>
                        <span class="badge">Explainable AI Toggle</span>
                        <span class="badge">Hackathon Demo Ready</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    start_clicked = st.button("Run Simulation", type="primary", use_container_width=True)
    replay_clicked = False
    if st.session_state.simulation_data is not None:
        replay_clicked = st.button("Replay", use_container_width=True)
    status_text = "Explainable AI enabled" if explainable_mode else "Black-box mode enabled"
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="status-label">Mode</div>
            <div class="status-value">{status_text}</div>
            <div class="status-strip">
                <div class="status-label">Demo Intent</div>
                <div class="status-value">Show judges how trust changes when reasoning is visible.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if start_clicked or replay_clicked:
    with st.spinner("Running market simulation..."):
        st.session_state.simulation_data = run_simulation(max_steps=max_steps, seed=seed)
    st.session_state.current_step = 0
    st.session_state.playing = True
    st.rerun()


data = st.session_state.simulation_data
if data is None:
    st.markdown(
        '<div class="footer-note">Run Simulation to launch the AI Trading Command Center.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


steps = data["steps"]
initial_value = data["initial_value"]
current_index = st.session_state.current_step if steps else 0

if st.session_state.playing and st.session_state.current_step == 0 and steps:
    st.session_state.current_step = 1
    current_index = 1

visible_step = steps[current_index - 1] if current_index else None

metric_row = st.columns(3, gap="medium")
if visible_step is None:
    portfolio_value = initial_value
    gain_pct = 0.0
    learning_score = 0.0
else:
    portfolio_value = visible_step["portfolio_value"]
    gain_pct = ((portfolio_value - initial_value) / initial_value) * 100
    learning_score = visible_step["learning_score"]

metrics = [
    ("Portfolio Value", f"${portfolio_value:,.2f}", "Current account value during the live run."),
    ("Gain %", f"{gain_pct:+.2f}%", f"Relative to the starting value of ${initial_value:,.2f}."),
    ("Learning Score", f"{learning_score:.3f}", "How well the agent is learning disciplined investing behavior."),
]

for col, metric in zip(metric_row, metrics):
    col.markdown(metric_card(*metric), unsafe_allow_html=True)


left_col, center_col, right_col = st.columns([1.05, 1.9, 1.1], gap="large")

with left_col:
    render_ai_panel(visible_step, explainable_mode, len(steps))

with center_col:
    st.markdown('<div class="section-title">Portfolio Command Chart</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">The main panel tracks the portfolio trajectory with a smooth Plotly line as the simulation unfolds.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.plotly_chart(
        build_portfolio_chart(steps, initial_value, current_index),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    progress = current_index / len(steps) if steps else 0.0
    st.progress(progress, text=f"Simulation Progress: {current_index}/{len(steps)} steps")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    render_feed(steps, current_index)


st.markdown(
    '<div class="footer-note">FINLEARN AI TUTOR • PREMIUM STREAMLIT DEMO • EXPLAINABLE INVESTING</div>',
    unsafe_allow_html=True,
)


if st.session_state.playing:
    if st.session_state.current_step < len(steps):
        sleep(speed_map[playback_speed])
        st.session_state.current_step += 1
        st.rerun()
    else:
        st.session_state.playing = False
