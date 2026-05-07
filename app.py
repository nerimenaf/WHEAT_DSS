from __future__ import annotations
from pathlib import Path
from datetime import date
import traceback
import pandas as pd
from dss.episodic_memory import load_episodes
import time
from dss.episodic_memory import start_episode, finalize_episode_outcome
from dss.local_memory_builder import rebuild_local_memory_indexes
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from dss.context import (
    build_shared_context, save_shared_context, load_shared_context,
    record_farmer_action, apply_irrigation_effect, apply_fertilization_effect
)
from dss.hybrid_retriever import HybridRAGStore, HybridRetriever
from dss.agents import IrrigationAgent, FertilizationAgent
from dss.llm import LlmSpec
from dss.orchestrator import orchestrate
from dss.config import settings


# ----------------------------
# Page + basic styling
# ----------------------------
st.set_page_config(page_title="Wheat DSS", layout="wide")

st.markdown(
    """
<style>
/* General */
.small-note { color: #6b7280; font-size: 0.9rem; }

/* Cards */
.card {
  border-radius: 14px;
  padding: 16px 18px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 1px 8px rgba(0,0,0,0.05);
  margin-bottom: 14px;
}
.card h3 { margin: 0 0 6px 0; }
.card p { margin: 6px 0; }

/* Decision colors */
.good { background: #ecfdf3; border-color: #a7f3d0; }
.warn { background: #fffbeb; border-color: #fde68a; }
.bad  { background: #fef2f2; border-color: #fecaca; }
.neutral { background: #eff6ff; border-color: #bfdbfe; }

/* Big action button */
div.stButton > button[kind="primary"] {
  border-radius: 12px;
  padding: 0.7rem 1rem;
  font-weight: 650;
}
.day-card{
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  margin-bottom: 12px;
  background: white;
}
.day-title{
  font-weight: 750;
  margin-bottom: 8px;
  font-size: 1.02rem;
}
.badge{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  margin: 6px 8px 0 0;
  font-size: 0.9rem;
  border: 1px solid rgba(0,0,0,0.06);
}
.badge-irrigation{ background:#eff6ff; border-color:#bfdbfe; }
.badge-fertilization{ background:#ecfdf3; border-color:#a7f3d0; }
.badge-warning{ background:#fffbeb; border-color:#fde68a; }
.badge-danger{ background:#fef2f2; border-color:#fecaca; }
.badge-neutral{ background:#f3f4f6; border-color:#e5e7eb; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Wheat Decision Support System")
st.caption("Farmer view (simple) + Developer view (full technical details)")

context_path = Path("shared_context.json")

def _task_to_badge(task: str) -> tuple[str, str]:
    t = (task or "").lower()

    if "irrigation:" in t or "irrigate" in t:
        return ("badge badge-irrigation", task)
    if "fertilization:" in t or "fertiliz" in t:
        return ("badge badge-fertilization", task)
    if "avoid" in t:
        return ("badge badge-danger", task)
    if "delay" in t or "heavy rain" in t or "reassess" in t:
        return ("badge badge-warning", task)

    return ("badge badge-neutral", task)


def render_weekly_schedule_cards(weekly_plan: list[dict]) -> None:
    for day in weekly_plan:
        date_str = day.get("date", "—")
        tasks = day.get("tasks", []) or []

        badges_html = []
        for task in tasks:
            cls, text = _task_to_badge(task)
            # escape minimal HTML special chars
            text = (text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"))
            badges_html.append(f'<span class="{cls}">{text}</span>')

        st.markdown(
            f"""
<div class="day-card">
  <div class="day-title">{date_str}</div>
  <div>{''.join(badges_html)}</div>
</div>
""",
            unsafe_allow_html=True,
        )
# ----------------------------
# Session state controls
# ----------------------------
if "run_dss" not in st.session_state:
    st.session_state.run_dss = False

# Apply pending widget updates BEFORE widgets are instantiated
if "pending_widget_updates" in st.session_state:
    updates = st.session_state.pop("pending_widget_updates")
    for k, v in updates.items():
        st.session_state[k] = v


# ----------------------------
# Load previous context (defaults)
# ----------------------------
prev_ctx = None
if context_path.exists():
    try:
        prev_ctx = load_shared_context(context_path)
    except Exception:
        prev_ctx = None

prev_field = (prev_ctx or {}).get("field", {})
prev_loc = (prev_field or {}).get("location", {})

default_field_name = prev_field.get("field_name", "Demo Field")
default_moist = float(prev_field.get("soil_moisture_pct", 22.0) or 22.0)
default_ph = float(prev_field.get("soil_ph", 6.8) or 6.8)
default_n = float(prev_field.get("n_ppm", 15.0) or 15.0)
default_p = float(prev_field.get("p_ppm", 9.0) or 9.0)
default_k = float(prev_field.get("k_ppm", 70.0) or 70.0)
default_lat = float(prev_loc.get("lat", 31.5204) or 31.5204)
default_lon = float(prev_loc.get("lon", 74.3587) or 74.3587)
default_soil_type = prev_field.get("soil_type", "Loamy")

try:
    default_sowing = date.fromisoformat(prev_field.get("sowing_date")) if prev_field.get("sowing_date") else date.today()
except Exception:
    default_sowing = date.today()

if "initialized_inputs" not in st.session_state:
    st.session_state.initialized_inputs = True
    st.session_state.field_name = default_field_name
    st.session_state.soil_moisture_pct = default_moist
    st.session_state.soil_ph = default_ph
    st.session_state.n_ppm = default_n
    st.session_state.p_ppm = default_p
    st.session_state.k_ppm = default_k
    st.session_state.sowing_date = default_sowing
    st.session_state.lat = default_lat
    st.session_state.lon = default_lon
    st.session_state.field_area_ha = float(prev_field.get("field_area_ha", 0.0) or 0.0)
    st.session_state.soil_type = default_soil_type


# ----------------------------
# Cached retrievers (faster)
# ----------------------------
@st.cache_resource
def load_hybrid_retrievers(use_reranker: bool = True):
    irrig_store = HybridRAGStore(Path("knowledge/index/irrigation"), use_reranker=use_reranker)
    fert_store = HybridRAGStore(Path("knowledge/index/fertilization"), use_reranker=use_reranker)

    irrigation_retriever = HybridRetriever(
        irrig_store, alpha=0.6, dense_k=20, sparse_k=20, final_k=5
    )
    fertilization_retriever = HybridRetriever(
        fert_store, alpha=0.6, dense_k=20, sparse_k=20, final_k=5
    )
    return irrigation_retriever, fertilization_retriever


# ----------------------------
# Helper UI functions
# ----------------------------
def decision_class(decision: str) -> str:
    d = (decision or "").lower()
    if any(x in d for x in ["do not", "hold", "no ", "avoid"]):
        return "good"
    if "delay" in d:
        return "warn"
    if any(x in d for x in ["irrigate", "apply fertilizer", "apply"]):
        return "neutral"
    return "neutral"


def render_action_card(title: str, rec) -> None:
    cls = decision_class(rec.decision)
    st.markdown(
        f"""
<div class="card {cls}">
  <h3>{title}</h3>
  <p><b>Decision:</b> {rec.decision}</p>
  <p><b>When:</b> {rec.timing}</p>
  <p><b>How much:</b> {getattr(rec, "quantity", "")}</p>
  <p><b>Why:</b> {rec.reason}</p>
  <p class="small-note"><b>Sources:</b> {", ".join(rec.citations) if rec.citations else "—"}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    


# ----------------------------
# Sidebar: Mode + Advanced settings
# ----------------------------
with st.sidebar:
    st.header("Mode")
    view_mode = st.radio("Choose interface", ["Farmer view", "Developer view"], index=0)

    st.divider()
    st.header("LLM (advanced)")
    st.caption("Farmer view hides these by default.")
    with st.expander("LLM settings", expanded=(view_mode == "Developer view")):
        backend = st.selectbox("Backend", ["ollama", "openai"], index=0 if settings.default_backend == "ollama" else 1)
        if backend == "openai":
            model = st.text_input("OpenAI model", value=settings.default_openai_model)
            st.write("Requires `OPENAI_API_KEY`.")
        else:
            model = st.text_input("Ollama model", value=getattr(settings, "default_ollama_model", "mistral"))
            st.write("Requires local Ollama running.")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

        use_reranker = st.checkbox("Use cross-encoder reranker (slower, better)", value=True)
    use_rag = st.checkbox("RAG (use KB retrieval)", value=True)
    memory_enabled = st.checkbox("Memory (retrieve past experiences)", value=True)
    memory_filtering_enabled = st.checkbox("Feedback filtering for memory (keep only good episodes)", value=True)

    st.divider()
    st.header("Experimental Mode")

    experimental_mode = st.checkbox("Enable Benchmark Mode")

    if experimental_mode:
        from dss.benchmark_scenarios import SCENARIOS

        scenario_id = st.selectbox(
            "Select Scenario",
            list(SCENARIOS.keys())
        )




# Defaults if expander not opened yet (avoid UnboundLocalError)
if "backend" not in locals():
    backend = "ollama"
if "model" not in locals():
    model = getattr(settings, "default_ollama_model", "mistral")
if "temperature" not in locals():
    temperature = 0.0
if "use_reranker" not in locals():
    use_reranker = True


# ----------------------------
# Inputs (Farmer-friendly)
# ----------------------------
st.subheader("1) Field information")

if view_mode == "Farmer view":
    st.info("Tip: Fill only what you know. Advanced fields are optional.", icon="ℹ️")

col1, col2, col3 = st.columns(3)

with col1:
    st.text_input("Field name", key="field_name")
    st.number_input("Soil moisture (%)", 0.0, 100.0, step=1.0, key="soil_moisture_pct")

with col2:
    st.number_input("Nitrogen N (ppm)", 0.0, 300.0, step=1.0, key="n_ppm")
    st.number_input("Phosphorus P (ppm)", 0.0, 200.0, step=1.0, key="p_ppm")
    st.number_input("Potassium K (ppm)", 0.0, 600.0, step=1.0, key="k_ppm")

with col3:
    st.date_input("Sowing date", key="sowing_date")

# Advanced inputs hidden for farmers
with st.expander("Optional details (location / pH / area)", expanded=(view_mode == "Developer view")):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Soil pH", 3.0, 10.0, step=0.1, key="soil_ph")
        st.number_input("Field area (ha)", 0.0, 1000.0, step=0.1, key="field_area_ha")
        st.selectbox("Soil type", ["Loamy", "Clayey", "Sandy", "Peaty", "Silty"], key="soil_type")
    with c2:
        st.number_input("Latitude", format="%.6f", key="lat")
    with c3:
        st.number_input("Longitude", format="%.6f", key="lon")

# ================================
# Experimental Mode (Benchmark)
# ================================
st.divider()
st.header("Experimental Mode")

experimental_mode = st.checkbox("Enable Benchmark Mode")

if experimental_mode:
    from dss.benchmark_scenarios import SCENARIOS

    scenario_id = st.selectbox(
        "Select Scenario",
        options=list(SCENARIOS.keys()),
        format_func=lambda x: f"{x} – {SCENARIOS[x]['name']}",
    )

    scenario = SCENARIOS[scenario_id]

    # ✅ Show scenario details (no rerun here)
    with st.expander("Scenario Details", expanded=True):
        st.write("Farm:", scenario["farm"])
        st.write("Climate:", scenario["climate"])
        st.write("Soil:", scenario["soil_type"])
        st.write("Stage:", scenario["stage"])
        st.write("Description:", scenario["description"])
        st.write("Evaluation Focus:", scenario["evaluation_focus"])
        st.write("Ground Truth:", scenario["ground_truth"])
        st.write("Time Series:")
        st.table(scenario["time_series"])

    # ✅ Load scenario safely
    if st.button("Load Scenario Into DSS"):
        st.session_state["pending_widget_updates"] = {
            "soil_type": scenario["soil_type"],
            "soil_moisture_pct": scenario["time_series"][0]["soil_moisture"],
            "n_ppm": 0,
            "p_ppm": 0,
            "k_ppm": 0,
        }

        # optional flag (not required but useful)
        st.session_state["scenario_loaded"] = True

        st.success("Scenario loaded. Now click 'Generate recommendations'.")
        st.rerun()


# ----------------------------
# Run DSS button
# ----------------------------
st.subheader("2) Get recommendations")

if st.button("Generate recommendations", type="primary"):
    st.session_state.run_dss = True
    st.session_state.run_token = time.time()
    st.session_state.episode_logged_for_token = None

if experimental_mode:
    if st.button("Run Experimental Scenario", type="primary"):
        st.session_state.run_dss = True




# ----------------------------
# Main DSS pipeline
# ----------------------------
if st.session_state.run_dss:
    if experimental_mode:
        scenario = SCENARIOS[scenario_id]
        scenario_ts = scenario["time_series"]
    else:
        scenario_ts = None

    ctx = build_shared_context(
        field_name=st.session_state.field_name,
        soil_moisture_pct=st.session_state.soil_moisture_pct,
        soil_ph=st.session_state.soil_ph,
        n_ppm=st.session_state.n_ppm,
        p_ppm=st.session_state.p_ppm,
        k_ppm=st.session_state.k_ppm,
        sowing_date=st.session_state.sowing_date,
        lat=st.session_state.lat,
        lon=st.session_state.lon,
        soil_type=st.session_state.soil_type,
        field_area_ha=st.session_state.field_area_ha,
    )
    if experimental_mode and scenario_ts:
            from datetime import date, timedelta

            daily_rain_override = {}

            for i, day in enumerate(scenario_ts):
                d = (date.today() + timedelta(days=i)).isoformat()
                daily_rain_override[d] = day.get("rain_mm", 0.0)

            ctx["weather"]["daily_rain_mm"] = daily_rain_override

            # Optional: override rain_next_24h_mm for day 1
            ctx["weather"]["rain_next_24h_mm"] = scenario_ts[0].get("rain_mm", 0.0)
    # Preserve actions from existing saved context
    if context_path.exists():
        try:
            old_ctx = load_shared_context(context_path)
            if old_ctx and "actions" in old_ctx:
                ctx["actions"] = old_ctx["actions"]
        except Exception:
            pass

    save_shared_context(ctx)
    # Local memory retrievers (optional; only if indexes exist)
    mem_irrig = None
    mem_fert = None

    mem_irrig_dir = Path("memory/index/irrigation")
    mem_fert_dir = Path("memory/index/fertilization")

    if (mem_irrig_dir / "dense.faiss").exists():
        mem_irrig_store = HybridRAGStore(mem_irrig_dir, use_reranker=False)
        mem_irrig = HybridRetriever(mem_irrig_store, alpha=0.6, dense_k=10, sparse_k=10, final_k=3)

    if (mem_fert_dir / "dense.faiss").exists():
        mem_fert_store = HybridRAGStore(mem_fert_dir, use_reranker=False)
        mem_fert = HybridRetriever(mem_fert_store, alpha=0.6, dense_k=10, sparse_k=10, final_k=3)

    # Quick farmer dashboard (weather + stage)
    a = ctx.get("analytics", {}) or {}
    w = ctx.get("weather", {}) or {}
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Growth stage", a.get("phenological_stage", "—"))
    top2.metric("Soil moisture", f"{ctx['field'].get('soil_moisture_pct', '—')}% ({a.get('soil_moisture_status','—')})")
    top3.metric("Rain next 24h", f"{w.get('rain_next_24h_mm', 0)} mm")
    top4.metric("Avg temp next 24h", f"{w.get('avg_temp_next_24h_c', 0)} °C")

    llm_spec = LlmSpec(backend=backend, model=model, temperature=temperature)

    # Retrievers (cached)
    irrigation_retriever, fertilization_retriever = load_hybrid_retrievers(use_reranker=use_reranker)

    # Run agents in parallel
    irrigation_rec = None
    fert_rec = None
    irr_error = None
    fert_error = None

    irrigation_agent = IrrigationAgent(irrigation_retriever, llm_spec, memory_retriever=mem_irrig)
    fert_agent = FertilizationAgent(fertilization_retriever, llm_spec, memory_retriever=mem_fert)
    orchestrated = None
    if experimental_mode and orchestrated is not None:
        scenario = SCENARIOS[scenario_id]
        gt = scenario["ground_truth"]

        # Extract DSS irrigation total
        dss_total = 0
        for day in orchestrated.weekly_plan:
            for task in day["tasks"]:
                if "Irrigation:" in task:
                    import re
                    match = re.search(r"(\d+\.?\d*)", task)
                    if match:
                        dss_total += float(match.group(1))

        # Evaluate
        within_range = (
            gt["irrigation_weekly_min"]
            <= dss_total
            <= gt["irrigation_weekly_max"]
        )

        st.subheader("Experimental Evaluation")

        st.write("Ground Truth Weekly Range:",
                f"{gt['irrigation_weekly_min']}–{gt['irrigation_weekly_max']} mm")

        st.write("DSS Weekly Irrigation:", f"{dss_total:.1f} mm")

        if within_range:
            st.success("Within expected irrigation range")
        else:
            st.error("Outside expected irrigation range")

        # Fertilization comparison
        st.write("Ground Truth NPK:", gt["N"], gt["P"], gt["K"])
        st.write("DSS Fertilization Decision:", fert_rec.quantity)
    

    with st.spinner("Analyzing field + weather..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_irr = executor.submit(irrigation_agent.run, ctx)
            fut_fert = executor.submit(fert_agent.run, ctx)

            try:
                irrigation_rec = fut_irr.result()
            except Exception as e:
                irr_error = e

            try:
                fert_rec = fut_fert.result()
            except Exception as e:
                fert_error = e

    st.subheader("3) Today’s actions")

    if view_mode == "Farmer view":
        if irr_error:
            st.error(f"Irrigation decision unavailable: {type(irr_error).__name__}: {irr_error}")
        else:
            render_action_card("Irrigation", irrigation_rec)

        if fert_error:
            st.error(f"Fertilization decision unavailable: {type(fert_error).__name__}: {fert_error}")
        else:
            render_action_card("Fertilization", fert_rec)

    else:
        # Developer view: keep JSON outputs visible
        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Irrigation Agent (JSON)")
            if irr_error:
                st.error(f"{type(irr_error).__name__}: {irr_error}")
            else:
                st.json(irrigation_rec.model_dump())
        with colB:
            st.markdown("### Fertilization Agent (JSON)")
            if fert_error:
                st.error(f"{type(fert_error).__name__}: {fert_error}")
            else:
                st.json(fert_rec.model_dump())
    
    if "episode_logged_for_token" not in st.session_state:
        st.session_state.episode_logged_for_token = None
    if "run_token" not in st.session_state:
        st.session_state.run_token = None
    if "last_episode_id" not in st.session_state:
        st.session_state.last_episode_id = None
###############################################################################################################################################

    st.subheader("4) Weekly plan")


    # 1) Compute orchestrated schedule
    if irrigation_rec is not None and fert_rec is not None:
        try:
            orchestrated = orchestrate(ctx, irrigation_rec, fert_rec)  # <-- call the function
        except Exception as e:
            st.error(f"Orchestrator failed: {type(e).__name__}: {e}")
            orchestrated = None
            st.write("DEBUG irrigation_rec:", irrigation_rec)
            st.write("DEBUG fert_rec:", fert_rec)
            st.text(traceback.format_exc())
            
    else:
        st.warning("Weekly plan not available because one or both agent outputs failed.")
    

    # 2) Display weekly plan as TABLE
    if orchestrated is not None:
        st.success(orchestrated.combined_notes)
        st.write("DEBUG irrigation_rec:", irrigation_rec)
        st.write("DEBUG fert_rec:", fert_rec)
        rows = [
            {
                "Date": d.get("date", ""),
                "Plan": " • ".join(d.get("tasks", []) or []),
            }
            for d in orchestrated.weekly_plan
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    if ctx.get("vwc_forecast"):
        st.subheader("Predicted Soil Moisture (VWC) – Next 7 Days")

        dates = [day["date"] for day in orchestrated.weekly_plan]

        df_vwc = pd.DataFrame({
            "Date": dates,
            "Predicted VWC (%)": ctx["vwc_forecast"]
        })

        st.table(df_vwc)

        # Developer-only JSON
        if view_mode == "Developer view":
            with st.expander("Full Orchestrator Output"):
                st.json(orchestrated.model_dump())

        # 3) Episodic memory logging (safe now)
        system_config = {
            "rag_enabled": use_rag,
            "memory_enabled": memory_enabled,
            "memory_filtering_enabled": memory_filtering_enabled,
            "retrieval": "hybrid+filter+rerank",
            "llm_backend": backend,
            "llm_model": model,
        }
        st.write("DEBUG irrigation_rec:", irrigation_rec)
        st.write("DEBUG fert_rec:", fert_rec)
        if st.session_state.episode_logged_for_token != st.session_state.run_token:
            eid = start_episode(
                ctx=ctx,
                irrigation_decision=irrigation_rec.model_dump(),
                fertilization_decision=fert_rec.model_dump(),
                schedule=orchestrated.model_dump(),
                system_config=system_config,
            )
            st.session_state.last_episode_id = eid
            st.session_state.episode_logged_for_token = st.session_state.run_token
    else:
        st.info("Skipping episode logging because orchestrated schedule is not available.")
########################################################################################################################################

    # Feedback (still available; can be simplified later)
    st.subheader("5) Farmer feedback (did you do the tasks?)")

    with st.form("feedback_form"):
        irrig_status = st.selectbox("Irrigation", ["No update", "Done", "Not done"])
        irrig_increase = st.slider("If done: increase soil moisture by (%)", 0.0, 50.0, 15.0, 1.0)
        irrig_notes = st.text_input("Irrigation notes", value="")

        fert_status = st.selectbox("Fertilization", ["No update", "Done", "Not done"])
        n_add = st.number_input("If done: add N (ppm)", value=0.0, step=1.0)
        p_add = st.number_input("If done: add P (ppm)", value=0.0, step=1.0)
        k_add = st.number_input("If done: add K (ppm)", value=0.0, step=1.0)
        fert_notes = st.text_input("Fertilization notes", value="")

        submit = st.form_submit_button("Save feedback")

        if submit:
            # 1) Load latest shared context
            ctx2 = load_shared_context(context_path)

            # 2) Apply farmer feedback (update actions + effects)
            if irrig_status == "Done":
                ctx2 = record_farmer_action(ctx2, action_type="irrigation", status="done", notes=irrig_notes)
                ctx2 = apply_irrigation_effect(ctx2, increase_pct=irrig_increase)
            elif irrig_status == "Not done":
                ctx2 = record_farmer_action(ctx2, action_type="irrigation", status="not_done", notes=irrig_notes)

            if fert_status == "Done":
                ctx2 = record_farmer_action(ctx2, action_type="fertilization", status="done", notes=fert_notes)
                ctx2 = apply_fertilization_effect(ctx2, n_add_ppm=n_add, p_add_ppm=p_add, k_add_ppm=k_add)
            elif fert_status == "Not done":
                ctx2 = record_farmer_action(ctx2, action_type="fertilization", status="not_done", notes=fert_notes)

            # 3) Save updated shared context
            save_shared_context(ctx2)

            # 4) Finalize outcome for the last episode (Outcome_t)
            eid = st.session_state.get("last_episode_id")
            if eid:
                finalize_episode_outcome(episode_id=eid, ctx_after=ctx2)

            # 5) Rebuild local memory indexes from high-quality episodes
            thr = 0.75 if memory_filtering_enabled else 0.0
            rebuild_local_memory_indexes(threshold=thr)

            # 6) Update UI widget values next rerun
            st.session_state["pending_widget_updates"] = {
                "soil_moisture_pct": float(ctx2["field"]["soil_moisture_pct"]),
                "n_ppm": float(ctx2["field"]["n_ppm"]),
                "p_ppm": float(ctx2["field"]["p_ppm"]),
                "k_ppm": float(ctx2["field"]["k_ppm"]),
            }

            st.session_state.run_dss = True
            st.success("Saved. Refreshing recommendations using updated data...")
            st.rerun()

    # Developer-only: show full context on demand
    with st.expander("Technical details (shared context + raw JSON)", expanded=(view_mode == "Developer view")):
        st.markdown("### Shared context (common memory)")
        st.json(ctx)
        st.write("### ML Prediction Info")
        if ctx.get("ml_layer"):
            st.write("### ML Prediction Info")
            st.json(ctx["ml_layer"])
        with st.expander("Experiments / Metrics (Developer)", expanded=False):
            eps = load_episodes()
            if not eps:
                st.info("No episodes recorded yet. Run recommendations and save feedback first.")
            else:
                rows = []
                for e in eps:
                    rows.append({
                        "time_utc": e.get("time_utc"),
                        "cvr": (e.get("constraints_t", {}) or {}).get("cvr"),
                        "feedback": (e.get("feedback_t", {}) or {}).get("score"),
                        "delta_moist": (e.get("outcome_t", {}) or {}).get("delta_soil_moisture_pct"),
                        "delta_n": (e.get("outcome_t", {}) or {}).get("delta_n_ppm"),
                        "delta_p": (e.get("outcome_t", {}) or {}).get("delta_p_ppm"),
                        "delta_k": (e.get("outcome_t", {}) or {}).get("delta_k_ppm"),
                    })

                df = pd.DataFrame(rows)
                st.write("CVR over time")
                st.line_chart(df.set_index("time_utc")["cvr"])

                st.write("Feedback score over time")
                st.line_chart(df.set_index("time_utc")["feedback"])

st.divider()
st.caption("Disclaimer: Prototype for learning. Validate thresholds and decisions locally with agronomists.")