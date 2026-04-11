import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from env.environment import FinLearnEnv
from env.models import Action
import time

# --- Page Config ---
st.set_page_config(
    page_title="FinLearn Tutor Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .regime-bull { background-color: #d4edda; color: #155724; }
    .regime-bear { background-color: #f8d7da; color: #721c24; }
    .regime-stable { background-color: #fff3cd; color: #856404; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'env' not in st.session_state:
    st.session_state.env = FinLearnEnv(max_steps=30, seed=42)
    st.session_state.obs_history = [st.session_state.env.state()]
    st.session_state.rewards_history = []
    st.session_state.actions_history = []
    st.session_state.done = False
    st.session_state.info_history = []

def reset_env():
    st.session_state.env.reset()
    st.session_state.obs_history = [st.session_state.env.state()]
    st.session_state.rewards_history = []
    st.session_state.actions_history = []
    st.session_state.done = False
    st.session_state.info_history = []

def run_step(action_id):
    if not st.session_state.done:
        obs, reward, done, info = st.session_state.env.step(action_id)
        st.session_state.obs_history.append(obs)
        st.session_state.rewards_history.append(reward.value)
        st.session_state.actions_history.append(action_id)
        st.session_state.info_history.append(info)
        st.session_state.done = done

# --- Header Section ---
st.title("FinLearn Tutor")
st.subheader("AI-Powered Retail Investor Decision Engine")

current_obs = st.session_state.obs_history[-1]

header_col1, header_col2, header_col3, header_col4 = st.columns(4)
with header_col1:
    st.metric("Current Step", f"{current_obs.step} / {st.session_state.env.max_steps}")
with header_col2:
    regime = current_obs.market_regime
    st.markdown(f"**Market Regime**")
    st.markdown(f'<span class="status-badge regime-{regime.lower()}">{regime}</span>', unsafe_allow_html=True)
with header_col3:
    st.markdown(f"**Risk Profile**")
    st.info(f"{current_obs.risk_appetite.capitalize()} Risk")
with header_col4:
    st.markdown(f"**Investment Goal**")
    st.info(current_obs.goal.replace('_', ' ').capitalize())

st.divider()

# --- Row 1: Key Metrics ---
row1_col1, row1_col2, row1_col3 = st.columns(3)

with row1_col1:
    st.metric("Portfolio Value", f"${current_obs.portfolio_value:,.2f}", 
              delta=f"{current_obs.portfolio_value - 1000:,.2f}" if current_obs.step > 0 else None)
    st.metric("Cash Balance", f"${current_obs.cash_balance:,.2f}")

with row1_col2:
    total_pnl = current_obs.portfolio_value - 1000
    st.metric("Profit / Loss", f"${total_pnl:,.2f}", delta=f"{(total_pnl/1000)*100:.2f}%")
    st.metric("Learning Score", f"{current_obs.learning_score:.2%}")

with row1_col3:
    st.metric("Portfolio Volatility", f"{current_obs.portfolio_volatility:.4f}")
    st.metric("Max Drawdown", f"{current_obs.max_drawdown:.2%}")

# --- Row 2: Performance & Allocation ---
row2_col1, row2_col2 = st.columns([0.7, 0.3])

with row2_col1:
    st.markdown("### 📈 Portfolio Performance")
    # Prepare data for line chart
    history_df = pd.DataFrame([
        {
            "Step": i, 
            "Portfolio Value": o.portfolio_value,
            "ALPHA": o.prices["ALPHA"],
            "BETA": o.prices["BETA"],
            "GAMMA": o.prices["GAMMA"]
        } for i, o in enumerate(st.session_state.obs_history)
    ])
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=history_df["Step"], y=history_df["Portfolio Value"], name="Portfolio Value", line=dict(color='gold', width=4)))
    for stock in ["ALPHA", "BETA", "GAMMA"]:
        fig_perf.add_trace(go.Scatter(x=history_df["Step"], y=history_df[stock], name=f"{stock} Price", line=dict(dash='dot')))
    
    fig_perf.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

with row2_col2:
    st.markdown("### 🥧 Asset Allocation")
    holdings = current_obs.holdings
    prices = current_obs.prices
    allocation_data = {s: holdings[s] * prices[s] for s in holdings}
    allocation_data["CASH"] = current_obs.cash_balance
    
    fig_pie = px.pie(
        names=list(allocation_data.keys()),
        values=list(allocation_data.values()),
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Row 3: Rewards & Risk ---
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.markdown("### 📊 Reward Trend")
    if st.session_state.rewards_history:
        reward_df = pd.DataFrame({
            "Step": range(1, len(st.session_state.rewards_history) + 1),
            "Step Reward": st.session_state.rewards_history,
            "Cumulative Reward": pd.Series(st.session_state.rewards_history).cumsum()
        })
        fig_reward = go.Figure()
        fig_reward.add_trace(go.Bar(x=reward_df["Step"], y=reward_df["Step Reward"], name="Step Reward"))
        fig_reward.add_trace(go.Scatter(x=reward_df["Step"], y=reward_df["Cumulative Reward"], name="Cumulative", yaxis="y2"))
        fig_reward.update_layout(
            yaxis2=dict(overlaying='y', side='right'),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_reward, use_container_width=True)
    else:
        st.info("No trades executed yet.")

with row3_col2:
    st.markdown("### ⚠️ Risk Metrics Panel")
    risk_container = st.container()
    with risk_container:
        r_c1, r_c2 = st.columns(2)
        r_c1.metric("Concentration Score", f"{current_obs.concentration_score:.4f}")
        r_c2.metric("Portfolio Vol", f"{current_obs.portfolio_volatility:.6f}")
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_obs.concentration_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Concentration Risk"},
            gauge = {
                'axis': {'range': [None, 1]},
                'steps' : [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.7], 'color': "orange"},
                    {'range': [0.7, 1], 'color': "red"}],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.8}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Row 4: Action Log ---
st.markdown("### 🧠 Action Log")
if st.session_state.info_history:
    log_data = []
    for i in range(len(st.session_state.info_history)):
        info = st.session_state.info_history[i]
        log_data.append({
            "Step": i + 1,
            "Action": info.get("action", "N/A"),
            "Reward": info.get("reward", 0.0),
            "Done": "Yes" if i == len(st.session_state.info_history)-1 and st.session_state.done else "No",
            "Explanation": info.get("reason", "N/A")
        })
    log_df = pd.DataFrame(log_data).sort_values("Step", ascending=False)
    st.dataframe(log_df, use_container_width=True, height=300)
else:
    st.info("Start taking actions to see the log.")

# --- Bottom Control Panel ---
st.divider()
st.markdown("### 🕹️ Control Panel")
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1, 1, 1])

ACTION_MAP = {
    "HOLD": 0,
    "BUY_ALPHA": 1,
    "BUY_BETA": 2,
    "BUY_GAMMA": 3,
    "SELL_ALPHA": 4,
    "SELL_BETA": 5,
    "SELL_GAMMA": 6,
    "REBALANCE": 7,
    "GET_HINT": 8
}

with ctrl_col1:
    selected_action_name = st.selectbox("Select Action", list(ACTION_MAP.keys()))

with ctrl_col2:
    if st.button("▶ Step", disabled=st.session_state.done):
        run_step(ACTION_MAP[selected_action_name])
        st.rerun()

with ctrl_col3:
    if st.button("🔄 Reset"):
        reset_env()
        st.rerun()

with ctrl_col4:
    if st.button("⚡ Run Full Episode", disabled=st.session_state.done):
        with st.status("Running simulation...", expanded=True) as status:
            while not st.session_state.done:
                # For auto-run, we might want a simple strategy or just random
                # Here we use 'HOLD' or the selected action for the whole run
                run_step(ACTION_MAP[selected_action_name])
                status.update(label=f"Step {st.session_state.env.step_count} completed...")
                time.sleep(0.1)
            status.update(label="Episode Complete!", state="complete")
        st.rerun()

# --- Optional Hint Display ---
if st.session_state.info_history and "hint" in st.session_state.info_history[-1]:
    st.sidebar.markdown("### 💡 AI Tutor Hint")
    st.sidebar.info(st.session_state.info_history[-1]["hint"])
