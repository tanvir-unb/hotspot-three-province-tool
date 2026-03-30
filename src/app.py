from __future__ import annotations

import html
import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PATHOGENS_LATEST = DATA_DIR / "scored_pathogens_latest_3prov.csv"
PATHOGENS_HISTORY = DATA_DIR / "scored_pathogens_history_3prov.csv"
CHEMICALS_LATEST = DATA_DIR / "scored_chemicals_latest_3prov.csv"
CHEMICALS_HISTORY = DATA_DIR / "scored_chemicals_history_3prov.csv"

CHOSEN_PROVINCES = ["Ontario", "British Columbia", "Quebec"]
PATHOGEN_LABEL_MAP = {"Low": 25.0, "Moderate": 55.0, "High": 85.0, "Non-detect": 0.0}
TREND_LABEL_MAP = {"Decreasing": 25.0, "No Change": 50.0, "Increasing": 75.0, "No Recent Data": 50.0}
ALERT_ORDER = ["Low", "Moderate", "High", "Very High"]

st.set_page_config(page_title="Hotspot Finder - 3 Province Scope", layout="wide")

if "selected_module" not in st.session_state:
    st.session_state["selected_module"] = "Overview"
if "module_radio" not in st.session_state:
    st.session_state["module_radio"] = st.session_state["selected_module"]
if "show_upload_options" not in st.session_state:
    st.session_state["show_upload_options"] = False
if "pending_module" in st.session_state:
    target_module = st.session_state.pop("pending_module")
    st.session_state["selected_module"] = target_module
    st.session_state["module_radio"] = target_module


@st.cache_data
def load_data():
    p_latest = pd.read_csv(PATHOGENS_LATEST, parse_dates=["weekstart"])
    p_hist = pd.read_csv(PATHOGENS_HISTORY, parse_dates=["weekstart"])
    c_latest = pd.read_csv(CHEMICALS_LATEST, parse_dates=["collection_date"], low_memory=False)
    c_hist = pd.read_csv(CHEMICALS_HISTORY, parse_dates=["collection_date"], low_memory=False)
    return p_latest, p_hist, c_latest, c_hist


def alert_band(score: float) -> str:
    if pd.isna(score):
        return "Unclassified"
    if score < 40:
        return "Low"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "High"
    return "Very High"


def index_for(options, value, fallback=0):
    try:
        return options.index(value)
    except Exception:
        return fallback


def render_css():
    st.markdown(
        """
        <style>
        .kpi-card {
            border: 1px solid #dbe5f1;
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            box-shadow: 0 6px 18px rgba(31, 78, 121, 0.06);
            min-height: 150px;
        }
        .kpi-card h4 {
            margin: 0 0 8px 0;
            font-size: 1.1rem;
            color: #12324a;
        }
        .kpi-big {
            font-size: 2rem;
            font-weight: 700;
            color: #0f5d9c;
            line-height: 1.1;
            margin-bottom: 8px;
        }
        .kpi-sub { color: #425466; font-size: 0.95rem; margin-bottom: 6px; }
        .kpi-meta { color: #5b6b7a; font-size: 0.85rem; }
        .chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; margin-bottom: 8px; }
        .chip {
            display: inline-block; padding: 6px 12px; border-radius: 999px;
            background: #edf5ff; border: 1px solid #d6e8ff; color: #174a7c; font-size: 0.86rem;
        }
        .alert-card {
            border: 1px solid #dbe5f1; border-left: 6px solid #0f5d9c; border-radius: 18px;
            padding: 18px; background: #ffffff; box-shadow: 0 6px 18px rgba(31, 78, 121, 0.06); min-height: 170px;
        }
        .alert-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 8px; color: #12324a; }
        .alert-score { font-size: 1.7rem; font-weight: 700; color: #b04700; margin: 2px 0 8px 0; }
        .helper-note { color: #5b6b7a; font-size: 0.92rem; }

        .metric-card {
            border: 1px solid #d8e2ec; border-radius: 18px; background: #f8fbff; padding: 14px 12px;
            min-height: 130px; text-align: center; box-shadow: 0 6px 16px rgba(31,78,121,0.05);
        }
        .metric-card .metric-label { font-size: 0.95rem; color: #4b6174; margin-bottom: 6px; font-weight: 600; }
        .metric-card .metric-value { font-size: 2rem; line-height: 1.05; font-weight: 800; color: #12324a; }
        .metric-card .metric-caption { font-size: 0.82rem; color: #6a7c8c; margin-top: 6px; }
        .metric-card.output-low { background: #ecfdf3; border: 2px solid #a7f3d0; }
        .metric-card.output-moderate { background: #fff7e6; border: 2px solid #fde68a; }
        .metric-card.output-high { background: #fff3e8; border: 2px solid #fdba74; }
        .metric-card.output-veryhigh { background: #fff1f2; border: 2px solid #fda4af; }
        .formula-symbol {
            font-size: 2rem; font-weight: 800; color: #78909c; text-align: center; padding-top: 38px;
        }
        .pathogen-table-wrap { border: 1px solid #dbe5f1; border-radius: 18px; overflow: hidden; background: white; }
        .pathogen-table { width: 100%; border-collapse: collapse; }
        .pathogen-table th {
            background: #f8fbff; color: #334e68; font-weight: 700; padding: 12px 10px; font-size: 0.92rem;
            border-bottom: 1px solid #e5edf5; text-align: left;
        }
        .pathogen-table td {
            padding: 11px 10px; border-bottom: 1px solid #eef3f8; font-size: 0.92rem; vertical-align: middle;
        }
        .pathogen-table tr:last-child td { border-bottom: none; }
        .pill {
            display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 0.8rem; font-weight: 700;
        }
        .pill-low { background: #dcfce7; color: #166534; }
        .pill-moderate { background: #fef3c7; color: #92400e; }
        .pill-high { background: #fee2e2; color: #991b1b; }
        .pill-veryhigh { background: #ef4444; color: white; }
        .pill-nondetect { background: #eef2ff; color: #3730a3; }
        .trend-icon { font-size: 1.15rem; line-height: 1; }
        .score-wrap { min-width: 160px; }
        .score-label-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .score-number { font-weight: 700; color: #12324a; }
        .score-track { width: 100%; height: 10px; background: #edf2f7; border-radius: 999px; overflow: hidden; }
        .score-fill { height: 100%; border-radius: 999px; }
        .small-muted { color: #6a7c8c; font-size: 0.84rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def simple_line_chart(df, date_col, metric_cols, title):
    chart_df = df[[date_col] + metric_cols].copy().set_index(date_col)
    st.line_chart(chart_df, height=320)
    st.caption(title)


def render_province_card(province, active_sites, top_score, chemical_series):
    st.markdown(
        f"""
        <div class="kpi-card">
            <h4>{province}</h4>
            <div class="kpi-big">{active_sites}</div>
            <div class="kpi-sub">Active monitoring sites</div>
            <div class="kpi-meta">Top pathogen hotspot: {top_score:.1f}</div>
            <div class="kpi-meta">Chemical series in scope: {chemical_series:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_card(title, line_1, score, line_2, line_3):
    st.markdown(
        f"""
        <div class="alert-card">
            <div class="alert-title">{title}</div>
            <div>{line_1}</div>
            <div class="alert-score">Hotspot score: {score:.1f}</div>
            <div>{line_2}</div>
            <div class="helper-note">{line_3}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hotspot_fill_color(score: float) -> str:
    if pd.isna(score):
        return "#94a3b8"
    if score < 40:
        return "#22c55e"
    if score < 60:
        return "#f59e0b"
    if score < 80:
        return "#f97316"
    return "#ef4444"


def output_class_for_score(score: float) -> str:
    band = alert_band(score)
    return {
        "Low": "output-low",
        "Moderate": "output-moderate",
        "High": "output-high",
        "Very High": "output-veryhigh",
    }.get(band, "")


def metric_card_html(label: str, value: float, caption: str = "", output_score: float | None = None) -> str:
    card_class = "metric-card"
    if output_score is not None:
        card_class += f" {output_class_for_score(output_score)}"
    return f"""
    <div class="{card_class}">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{value:.1f}</div>
        <div class="metric-caption">{html.escape(caption)}</div>
    </div>
    """


def pill_html(value: str) -> str:
    val = str(value)
    key = val.lower().replace(" ", "")
    cls = {
        "low": "pill-low",
        "moderate": "pill-moderate",
        "high": "pill-high",
        "veryhigh": "pill-veryhigh",
        "non-detect": "pill-nondetect",
        "nondetect": "pill-nondetect",
    }.get(key, "pill-low")
    return f'<span class="pill {cls}">{html.escape(val)}</span>'


def trend_icon(value: str) -> str:
    val = str(value)
    if val == "Increasing":
        return '<span class="trend-icon" title="Increasing">⬆️</span>'
    if val == "Decreasing":
        return '<span class="trend-icon" title="Decreasing">⬇️</span>'
    return '<span class="trend-icon" title="No Change">➡️</span>'


def chemical_trend_icon(row: pd.Series) -> str:
    pct = row.get("trend_pct_change")
    if pd.isna(pct):
        return '<span class="trend-icon" title="Flat">➡️</span>'
    if float(pct) > 10:
        return '<span class="trend-icon" title="Rising">⬆️</span>'
    if float(pct) < -10:
        return '<span class="trend-icon" title="Falling">⬇️</span>'
    return '<span class="trend-icon" title="Flat">➡️</span>'


def chemical_facility_label(row: pd.Series) -> str:
    province = str(row.get("province_scope") or "").strip()
    water = str(row.get("receiving_water") or "").strip()
    if province and water and water.lower() != 'nan':
        return f"{province} • {water}"
    if province:
        return province
    if water and water.lower() != 'nan':
        return water
    return "Facility context unavailable"


def chemical_facility_tooltip(row: pd.Series) -> str:
    bits = []
    if pd.notna(row.get("wwtp_code")):
        bits.append(f"WWTP code: {row['wwtp_code']}")
    if pd.notna(row.get("province_scope")):
        bits.append(f"Province scope: {row['province_scope']}")
    if pd.notna(row.get("receiving_water")):
        bits.append(f"Receiving water: {row['receiving_water']}")
    if pd.notna(row.get("liquid_treatment_type")):
        bits.append(f"Treatment: {row['liquid_treatment_type']}")
    return html.escape(" | ".join(bits) if bits else "Facility context unavailable")


def hotspot_bar_html(score: float, max_score: float = 100.0) -> str:
    fill_pct = min(max((score / max_score) * 100.0, 0.0), 100.0)
    fill_color = hotspot_fill_color(score)
    return f"""
    <div class="score-wrap">
        <div class="score-label-row">
            <span class="score-number">{score:.1f}</span>
            <span class="small-muted">/ 100</span>
        </div>
        <div class="score-track">
            <div class="score-fill" style="width:{fill_pct:.1f}%; background:{fill_color};"></div>
        </div>
    </div>
    """


def render_pathogen_kpi_equation(row: pd.Series):
    cols = st.columns([1.45, 0.16, 1.45, 0.16, 1.45, 0.16, 1.45, 0.20, 1.75])
    with cols[0]:
        st.markdown(metric_card_html("Level", float(row["level_score"]), "How high now?"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card_html("Trend", float(row["trend_score"]), "Is it rising?"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_card_html("Jump", float(row["jump_score"]), "Latest change"), unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[6]:
        st.markdown(metric_card_html("Reliability", float(row["reliability_score"]), "Trust in record"), unsafe_allow_html=True)
    with cols[7]:
        st.markdown('<div class="formula-symbol">=</div>', unsafe_allow_html=True)
    with cols[8]:
        st.markdown(
            metric_card_html("HOTSPOT", float(row["hotspot_score"]), f"{alert_band(float(row['hotspot_score']))} band", output_score=float(row["hotspot_score"])),
            unsafe_allow_html=True,
        )
    st.caption("Visual logic: Level, Trend, and Jump are combined, then the result is moderated by Reliability to produce the final Hotspot score.")


def render_pathogen_history_chart(hist: pd.DataFrame):
    display_map = {
        "current_value": "Current Value",
        "level_score": "Level Score",
        "trend_score": "Trend Score",
        "jump_score": "Jump Score",
        "reliability_score": "Reliability Score",
        "hotspot_score": "Hotspot Score",
    }
    reverse_map = {v: k for k, v in display_map.items()}
    metric_options = list(reverse_map.keys())

    default_metric_1 = "Hotspot Score"
    default_metric_2 = "Reliability Score"

    col_a, col_b = st.columns(2)
    with col_a:
        metric_1 = st.selectbox(
            "Metric 1",
            metric_options,
            index=metric_options.index(default_metric_1),
            key="pathogen_hist_metric_1",
        )
    metric_2_options = [m for m in metric_options if m != metric_1]
    with col_b:
        fallback_index = metric_2_options.index(default_metric_2) if default_metric_2 in metric_2_options else 0
        metric_2 = st.selectbox(
            "Metric 2",
            metric_2_options,
            index=fallback_index,
            key="pathogen_hist_metric_2",
        )

    selected_pretty = [metric_1, metric_2]
    selected_cols = [reverse_map[m] for m in selected_pretty]

    plot_df = hist[["weekstart"] + selected_cols].copy().rename(columns=display_map)
    long_df = plot_df.melt("weekstart", var_name="Metric", value_name="Value")

    color_lookup = {
        "Current Value": "#64748b",
        "Hotspot Score": "#0f5d9c",
        "Reliability Score": "#16a34a",
        "Level Score": "#8b5cf6",
        "Trend Score": "#f59e0b",
        "Jump Score": "#ef4444",
    }
    color_domain = selected_pretty
    color_range = [color_lookup[m] for m in selected_pretty]

    base = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.8)
        .encode(
            x=alt.X("weekstart:T", title="Week"),
            y=alt.Y("Value:Q", title="Score / value"),
            color=alt.Color(
                "Metric:N",
                title="Metric",
                sort=selected_pretty,
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(orient="top", columns=2, symbolSize=170),
            ),
            tooltip=[
                alt.Tooltip("weekstart:T", title="Week"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Value:Q", title="Value", format=".1f"),
            ],
        )
        .properties(height=380)
    )

    threshold_df = pd.DataFrame({"Threshold": [88.0]})
    rule = alt.Chart(threshold_df).mark_rule(color="#ef4444", strokeDash=[7, 5], strokeWidth=2).encode(y="Threshold:Q")
    label_df = pd.DataFrame({"weekstart": [hist["weekstart"].max()], "Threshold": [88.0], "Label": ["Critical Hotspot Threshold"]})
    rule_label = (
        alt.Chart(label_df)
        .mark_text(color="#ef4444", align="right", baseline="bottom", dx=-4, dy=-4, fontWeight="bold")
        .encode(x="weekstart:T", y="Threshold:Q", text="Label:N")
    )

    chart = base + rule + rule_label
    st.altair_chart(chart, use_container_width=True)
    st.caption("Select any two metrics above to compare them directly in the same historical chart. The red rule marks the critical hotspot threshold at 88.")


def render_pathogen_table(df: pd.DataFrame):
    show = df[["city", "Location", "current_value", "latestLevel", "latestTrend", "hotspot_score", "alert_band"]].copy()
    show = show.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    rows_html = []
    for _, r in show.iterrows():
        city = "" if pd.isna(r["city"]) else html.escape(str(r["city"]))
        site = "" if pd.isna(r["Location"]) else html.escape(str(r["Location"]))
        current_value = "" if pd.isna(r["current_value"]) else f"{float(r['current_value']):.3f}"
        rows_html.append(
            f"""
            <tr>
                <td>{city}</td>
                <td>{site}</td>
                <td>{current_value}</td>
                <td>{pill_html(str(r['latestLevel']))}</td>
                <td>{trend_icon(str(r['latestTrend']))}</td>
                <td>{hotspot_bar_html(float(r['hotspot_score']))}</td>
                <td>{pill_html(str(r['alert_band']))}</td>
            </tr>
            """
        )
    table_html = f"""
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: white; }}
    .pathogen-table-wrap {{ border: 1px solid #dbe5f1; border-radius: 18px; overflow: hidden; background: white; }}
    .pathogen-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    .pathogen-table th {{ background: #f8fbff; color: #334e68; font-weight: 700; padding: 12px 10px; font-size: 0.92rem; border-bottom: 1px solid #e5edf5; text-align: left; }}
    .pathogen-table td {{ padding: 11px 10px; border-bottom: 1px solid #eef3f8; font-size: 0.92rem; vertical-align: middle; color: #12324a; }}
    .pathogen-table tr:last-child td {{ border-bottom: none; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 0.8rem; font-weight: 700; }}
    .pill-low {{ background: #dcfce7; color: #166534; }}
    .pill-moderate {{ background: #fef3c7; color: #92400e; }}
    .pill-high {{ background: #fee2e2; color: #991b1b; }}
    .pill-veryhigh {{ background: #ef4444; color: white; }}
    .pill-nondetect {{ background: #eef2ff; color: #3730a3; }}
    .trend-icon {{ font-size: 1.15rem; line-height: 1; }}
    .score-wrap {{ min-width: 160px; }}
    .score-label-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
    .score-number {{ font-weight: 700; color: #12324a; }}
    .score-track {{ width: 100%; height: 10px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .score-fill {{ height: 100%; border-radius: 999px; }}
    .small-muted {{ color: #6a7c8c; font-size: 0.84rem; }}
    </style>
    </head>
    <body>
    <div class="pathogen-table-wrap">
        <table class="pathogen-table">
            <thead>
                <tr>
                    <th style="width:14%">City</th>
                    <th style="width:20%">Site</th>
                    <th style="width:12%">Current value</th>
                    <th style="width:14%">Official level</th>
                    <th style="width:10%">Trend</th>
                    <th style="width:20%">Hotspot score</th>
                    <th style="width:10%">Alert band</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    height = min(640, 74 + 52 * len(show))
    components.html(table_html, height=height, scrolling=True)
    st.caption("Status pills show the official level label and the final hotspot band. Trend icons: ⬆️ increasing, ➡️ no change, ⬇️ decreasing. The hotspot bar shows the hotspot score on its full 0–100 scale.")



def alert_pill_html(value: str) -> str:
    val = str(value)
    band = val.lower().replace(" ", "")
    emoji = {
        "low": "🟢",
        "moderate": "🟡",
        "high": "🔴",
        "veryhigh": "🔴",
    }.get(band, "⚪")
    return pill_html(f"{emoji} {val}")


def render_chemical_kpi_equation(row: pd.Series):
    cols = st.columns([1.35, 0.14, 1.35, 0.14, 1.35, 0.14, 1.35, 0.18, 1.9])
    with cols[0]:
        st.markdown(metric_card_html("Level", float(row["level_score"]), "Relative concentration"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card_html("Trend", float(row["trend_score"]), "Recent direction"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_card_html("Jump", float(row["jump_score"]), "Latest change"), unsafe_allow_html=True)
    with cols[5]:
        st.markdown('<div class="formula-symbol">+</div>', unsafe_allow_html=True)
    with cols[6]:
        st.markdown(metric_card_html("Reliability", float(row["reliability_score"]), "Trust in record"), unsafe_allow_html=True)
    with cols[7]:
        st.markdown('<div class="formula-symbol">=</div>', unsafe_allow_html=True)
    with cols[8]:
        st.markdown(metric_card_html("HOTSPOT", float(row["hotspot_score"]), f"{alert_band(float(row['hotspot_score']))} band", output_score=float(row["hotspot_score"])), unsafe_allow_html=True)
    st.caption("Visual logic: Level, Trend, and Jump are combined, then the result is moderated by Reliability to produce the final Hotspot score.")


def render_chemical_predictive_chart(hist: pd.DataFrame, key_prefix: str = 'chem'):
    st.markdown("### Historical trajectory")
    display_map = {
        "current_value": "Current Value",
        "level_score": "Level Score",
        "trend_score": "Trend Score",
        "jump_score": "Jump Score",
        "reliability_score": "Reliability Score",
        "hotspot_score": "Hotspot Score",
    }
    reverse_map = {v: k for k, v in display_map.items()}
    metric_options = list(reverse_map.keys())

    default_metric_1 = "Hotspot Score"
    default_metric_2 = "Reliability Score"

    col_a, col_b = st.columns(2)
    with col_a:
        metric_1 = st.selectbox(
            "Metric 1",
            metric_options,
            index=metric_options.index(default_metric_1),
            key=f"{key_prefix}_hist_metric_1",
        )
    metric_2_options = [m for m in metric_options if m != metric_1]
    with col_b:
        fallback_index = metric_2_options.index(default_metric_2) if default_metric_2 in metric_2_options else 0
        metric_2 = st.selectbox(
            "Metric 2",
            metric_2_options,
            index=fallback_index,
            key=f"{key_prefix}_hist_metric_2",
        )

    selected_pretty = [metric_1, metric_2]
    selected_cols = [reverse_map[m] for m in selected_pretty]

    plot_df = hist[["collection_date"] + selected_cols].copy().rename(columns=display_map)
    long_df = plot_df.melt("collection_date", var_name="Metric", value_name="Value")

    color_lookup = {
        "Current Value": "#64748b",
        "Hotspot Score": "#0f5d9c",
        "Reliability Score": "#16a34a",
        "Level Score": "#8b5cf6",
        "Trend Score": "#f59e0b",
        "Jump Score": "#ef4444",
    }
    color_domain = selected_pretty
    color_range = [color_lookup[m] for m in selected_pretty]

    base = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.8)
        .encode(
            x=alt.X("collection_date:T", title="Date"),
            y=alt.Y("Value:Q", title="Score / value"),
            color=alt.Color(
                "Metric:N",
                title="Metric",
                sort=selected_pretty,
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(orient="top", columns=2, symbolSize=170),
            ),
            tooltip=[
                alt.Tooltip("collection_date:T", title="Date"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Value:Q", title="Value", format=".1f"),
            ],
        )
        .properties(height=380)
    )

    threshold_df = pd.DataFrame({"Threshold": [60.0]})
    rule = alt.Chart(threshold_df).mark_rule(color="#ef4444", strokeDash=[7, 5], strokeWidth=2).encode(y="Threshold:Q")
    label_df = pd.DataFrame({"collection_date": [hist["collection_date"].max()], "Threshold": [60.0], "Label": ["Action Threshold"]})
    rule_label = (
        alt.Chart(label_df)
        .mark_text(color="#ef4444", align="right", baseline="bottom", dx=-4, dy=-4, fontWeight="bold")
        .encode(x="collection_date:T", y="Threshold:Q", text="Label:N")
    )

    chart = base + rule + rule_label
    st.altair_chart(chart, use_container_width=True)
    st.caption("Select any two metrics above to compare them directly in the same historical chart. The red rule marks the action threshold at 60.")


def render_chemical_table(df: pd.DataFrame):
    show = df[[
        "province_scope", "wwtp_code", "receiving_water", "liquid_treatment_type",
        "analyte_name", "current_value", "detection_limit", "non_detect_rate_day",
        "trend_pct_change", "hotspot_score", "alert_band"
    ]].copy()
    show = show.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    rows_html = []
    for _, r in show.iterrows():
        facility_display = html.escape(chemical_facility_label(r))
        facility_tooltip = chemical_facility_tooltip(r)
        analyte = "" if pd.isna(r["analyte_name"]) else html.escape(str(r["analyte_name"]))

        is_below_dl = False
        try:
            is_below_dl = pd.notna(r.get("non_detect_rate_day")) and float(r.get("non_detect_rate_day", 0)) >= 1.0
        except Exception:
            is_below_dl = False
        if is_below_dl:
            current_value = "<strong>&lt; DL</strong>"
        elif pd.isna(r["current_value"]):
            current_value = "—"
        else:
            current_value = f"{float(r['current_value']):.3f}"

        detection_limit = "—" if pd.isna(r["detection_limit"]) else f"{float(r['detection_limit']):.3f}"
        rows_html.append(
            f"""
            <tr>
                <td><span title="{facility_tooltip}">{facility_display}</span></td>
                <td>{analyte}</td>
                <td style='text-align:center;'>{current_value}</td>
                <td style='text-align:center;'>{detection_limit}</td>
                <td style='text-align:center;'>{chemical_trend_icon(r)}</td>
                <td>{hotspot_bar_html(float(r['hotspot_score']))}</td>
                <td>{alert_pill_html(str(r['alert_band']))}</td>
            </tr>
            """
        )
    table_html = f"""
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
    body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: white; }}
    .chem-table-wrap {{ border: 1px solid #dbe5f1; border-radius: 18px; overflow: hidden; background: white; }}
    .chem-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    .chem-table th {{ background: #f8fbff; color: #334e68; font-weight: 700; padding: 12px 10px; font-size: 0.92rem; border-bottom: 1px solid #e5edf5; text-align: left; }}
    .chem-table td {{ padding: 11px 10px; border-bottom: 1px solid #eef3f8; font-size: 0.92rem; vertical-align: middle; color: #12324a; }}
    .chem-table tr:last-child td {{ border-bottom: none; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 4px 10px; font-size: 0.8rem; font-weight: 700; }}
    .pill-low {{ background: #dcfce7; color: #166534; }}
    .pill-moderate {{ background: #fef3c7; color: #92400e; }}
    .pill-high {{ background: #fee2e2; color: #991b1b; }}
    .pill-veryhigh {{ background: #ef4444; color: white; }}
    .score-wrap {{ min-width: 160px; }}
    .score-label-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }}
    .score-number {{ font-weight: 700; color: #12324a; }}
    .score-track {{ width: 100%; height: 10px; background: #edf2f7; border-radius: 999px; overflow: hidden; }}
    .score-fill {{ height: 100%; border-radius: 999px; }}
    .small-muted {{ color: #6a7c8c; font-size: 0.84rem; }}
    .trend-icon {{ font-size: 1.15rem; line-height: 1; }}
    </style>
    </head>
    <body>
    <div class="chem-table-wrap">
        <table class="chem-table">
            <thead>
                <tr>
                    <th style="width:22%">Facility / geography</th>
                    <th style="width:24%">Analyte</th>
                    <th style="width:10%">Current value</th>
                    <th style="width:11%">Detection limit</th>
                    <th style="width:8%">Trend</th>
                    <th style="width:17%">Hotspot score</th>
                    <th style="width:8%">Alert band</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    </body>
    </html>
    """
    height = min(700, 74 + 52 * len(show))
    components.html(table_html, height=height, scrolling=True)
    st.caption("Facility labels expand the raw WWTP codes using province and receiving-water context from the processed reference data; hover to see the original code and treatment details. A bold < DL means the measured value was below the detection limit. Trend arrows show rising, flat, or falling behaviour. Hotspot bars use the full 0–100 scale.")


def render_chemical_upload_popover():
    with st.popover("＋ Upload Custom CSV", use_container_width=True):
        st.caption("Paste CSV with at least: collection_date,current_value. Optional columns: detection_limit,is_non_detect,has_lab_flag.")
        sample = """collection_date,current_value,detection_limit,is_non_detect,has_lab_flag
2024-01-01,2.0,1.0,False,False
2024-03-01,4.0,1.0,False,False
2024-05-01,5.0,1.0,False,False
2024-07-01,4.0,1.0,False,False
2024-09-01,12.0,1.0,False,False"""
        pasted = st.text_area("Paste chemical CSV", value="", height=150, placeholder=sample, key="chemical_paste_text")
        if pasted.strip():
            try:
                pasted_df = pd.read_csv(io.StringIO(pasted))
                hist2, latest2 = score_chemical_series(pasted_df)
                st.success(f"Preview hotspot: {latest2['hotspot_score']:.1f} ({alert_band(float(latest2['hotspot_score']))})")
                render_chemical_kpi_equation(latest2)
                render_chemical_predictive_chart(hist2, key_prefix='chem_upload')
                st.dataframe(hist2.tail(10), use_container_width=True, height=220)
            except Exception as exc:
                st.error(f"Could not parse or score the pasted chemical CSV: {exc}")

def render_pathogen_upload_popover():
    with st.popover("Upload Custom CSV", use_container_width=True):
        st.caption("Paste CSV with at least: weekstart,current_value. Optional columns: latestLevel,latestTrend,populationcoverage.")
        sample = "weekstart,current_value,latestLevel,latestTrend,populationcoverage\n2026-01-25,2,Moderate,No Change,1.0\n2026-02-01,4,Moderate,No Change,1.0\n2026-02-08,5,Moderate,Increasing,1.0\n2026-02-15,4,Moderate,Increasing,1.0\n2026-02-22,12,High,Increasing,1.0"
        pasted = st.text_area("Paste pathogen CSV", value="", height=150, placeholder=sample, key="pathogen_paste_text")
        if pasted.strip():
            try:
                pasted_df = pd.read_csv(io.StringIO(pasted))
                hist2, latest2 = score_pathogen_series(pasted_df)
                st.success(f"Preview hotspot: {latest2['hotspot_score']:.1f} ({alert_band(float(latest2['hotspot_score']))})")
                mcols = st.columns(5)
                mcols[0].metric("Level", f"{latest2['level_score']:.1f}")
                mcols[1].metric("Trend", f"{latest2['trend_score']:.1f}")
                mcols[2].metric("Jump", f"{latest2['jump_score']:.1f}")
                mcols[3].metric("Reliability", f"{latest2['reliability_score']:.1f}")
                mcols[4].metric("Hotspot", f"{latest2['hotspot_score']:.1f}")
                st.dataframe(hist2.tail(10), use_container_width=True, height=220)
            except Exception as exc:
                st.error(f"Could not parse or score the pasted pathogen CSV: {exc}")


def go_to_pathogens(prefill=None, paste=False):
    st.session_state["pending_module"] = "Pathogens"
    if prefill is not None:
        st.session_state["pathogen_prefill"] = prefill
    if paste:
        st.session_state["upload_target"] = "Pathogens"
    st.rerun()


def go_to_chemicals(prefill=None, paste=False):
    st.session_state["pending_module"] = "Chemicals"
    if prefill is not None:
        st.session_state["chemical_prefill"] = prefill
    if paste:
        st.session_state["upload_target"] = "Chemicals"
    st.rerun()


def score_pathogen_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.copy()
    x["weekstart"] = pd.to_datetime(x["weekstart"])
    x["current_value"] = pd.to_numeric(x["current_value"], errors="coerce")
    x["populationcoverage"] = pd.to_numeric(x.get("populationcoverage"), errors="coerce")
    x = x.dropna(subset=["weekstart", "current_value"]).sort_values("weekstart").reset_index(drop=True)
    calc_rows = []
    vals = x["current_value"].to_numpy(dtype=float)
    for i in range(len(x)):
        recent = vals[max(0, i - 11): i + 1]
        current = vals[i]
        level_pct = 100.0 * float(np.sum(recent <= current)) / float(len(recent))
        level_off = PATHOGEN_LABEL_MAP.get(str(x.loc[i, "latestLevel"]).strip(), 50.0) if "latestLevel" in x.columns else 50.0
        level_score = (level_pct + level_off) / 2.0

        recent2 = recent[-2:]
        earlier = recent[max(0, len(recent) - 6): max(0, len(recent) - 2)]
        if len(earlier) > 0 and float(np.mean(earlier)) > 0:
            trend_pct = 100.0 * (float(np.mean(recent2)) - float(np.mean(earlier))) / float(np.mean(earlier))
            raw_trend = float(np.clip(50.0 + 1.2 * trend_pct, 0.0, 100.0))
        else:
            trend_pct = np.nan
            raw_trend = 50.0
        trend_off = TREND_LABEL_MAP.get(str(x.loc[i, "latestTrend"]).strip(), 50.0) if "latestTrend" in x.columns else 50.0
        trend_score = (raw_trend + trend_off) / 2.0

        if i > 0 and vals[i - 1] > 0:
            jump_pct = 100.0 * (current - vals[i - 1]) / vals[i - 1]
            jump_score = float(np.clip(50.0 + 2.0 * jump_pct, 0.0, 100.0))
        else:
            jump_pct = np.nan
            jump_score = 50.0

        k = len(recent)
        base_rel = 88.0 if k >= 6 else (78.0 if k == 5 else 65.0)
        cov = x.loc[i, "populationcoverage"] if "populationcoverage" in x.columns else np.nan
        if pd.isna(cov):
            cov_adj = -10.0
        elif cov >= 0.9:
            cov_adj = 5.0
        elif cov >= 0.5:
            cov_adj = 0.0
        else:
            cov_adj = -5.0
        rel = float(np.clip(base_rel + cov_adj + 0.0, 0.0, 100.0))
        blend = 0.45 * level_score + 0.35 * trend_score + 0.20 * jump_score
        mult = 0.7 + 0.3 * (rel / 100.0)
        hotspot = blend * mult

        calc_rows.append(
            {
                "weekstart": x.loc[i, "weekstart"],
                "current_value": current,
                "level_score": level_score,
                "trend_score": trend_score,
                "jump_score": jump_score,
                "reliability_score": rel,
                "hotspot_score": hotspot,
                "alert_band": alert_band(hotspot),
            }
        )
    hist = pd.DataFrame(calc_rows)
    latest = hist.iloc[-1]
    return hist, latest


def score_chemical_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.copy()
    x["collection_date"] = pd.to_datetime(x["collection_date"])
    for col in ["current_value", "detection_limit"]:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    if "is_non_detect" not in x.columns:
        x["is_non_detect"] = False
    if "has_lab_flag" not in x.columns:
        x["has_lab_flag"] = False
    x = x.dropna(subset=["collection_date", "current_value"]).sort_values("collection_date").reset_index(drop=True)
    vals = np.where(x["is_non_detect"].fillna(False), x["detection_limit"].fillna(0) * 0.5, x["current_value"]).astype(float)
    days = x["collection_date"].to_numpy(dtype="datetime64[D]").astype("int64")
    global_max = days.max()
    calc_rows = []
    for i in range(len(x)):
        start = max(0, i - 11)
        recent = vals[start : i + 1]
        current = vals[i]
        level_pct = 100.0 * float(np.sum(recent <= current)) / float(len(recent))
        level_score = level_pct
        recent2 = recent[-2:]
        earlier = recent[max(0, len(recent) - 6): max(0, len(recent) - 2)]
        if len(earlier) > 0 and float(np.mean(earlier)) > 0:
            trend_pct = 100.0 * (float(np.mean(recent2)) - float(np.mean(earlier))) / float(np.mean(earlier))
            trend_score = float(np.clip(50.0 + 1.2 * trend_pct, 0.0, 100.0))
        else:
            trend_pct = np.nan
            trend_score = 50.0
        if i > 0 and vals[i - 1] > 0:
            jump_pct = 100.0 * (current - vals[i - 1]) / vals[i - 1]
            jump_score = float(np.clip(50.0 + 2.0 * jump_pct, 0.0, 100.0))
        else:
            jump_pct = np.nan
            jump_score = 50.0
        k = len(recent)
        base_rel = 88.0 if k >= 6 else (78.0 if k == 5 else 65.0)
        age = float(global_max - days[i])
        rec_adj = 5.0 if age <= 365 else (0.0 if age <= 1095 else -8.0)
        nd_rate = float(x.loc[start:i, "is_non_detect"].fillna(False).astype(float).mean())
        flag_rate = float(x.loc[start:i, "has_lab_flag"].fillna(False).astype(float).mean())
        nd_adj = 5.0 if nd_rate <= 0.20 else (0.0 if nd_rate <= 0.50 else -5.0)
        fl_adj = 0.0 if flag_rate <= 0.20 else (-3.0 if flag_rate <= 0.50 else -6.0)
        recent_days = days[start : i + 1]
        med_gap = float(np.median(np.diff(recent_days))) if len(recent_days) > 1 else np.nan
        sp_adj = -3.0 if (k < 3 or np.isnan(med_gap)) else (3.0 if med_gap <= 45 else (0.0 if med_gap <= 180 else -5.0))
        rel = float(np.clip(base_rel + rec_adj + nd_adj + fl_adj + sp_adj, 0.0, 100.0))
        blend = 0.45 * level_score + 0.35 * trend_score + 0.20 * jump_score
        mult = 0.7 + 0.3 * (rel / 100.0)
        hotspot = blend * mult
        calc_rows.append(
            {
                "collection_date": x.loc[i, "collection_date"],
                "current_value": current,
                "level_score": level_score,
                "trend_score": trend_score,
                "jump_score": jump_score,
                "reliability_score": rel,
                "hotspot_score": hotspot,
                "non_detect_rate_recent": nd_rate,
                "flag_rate_recent": flag_rate,
                "alert_band": alert_band(hotspot),
            }
        )
    hist = pd.DataFrame(calc_rows)
    latest = hist.iloc[-1]
    return hist, latest


render_css()
p_latest, p_hist, c_latest, c_hist = load_data()
chemical_families = sorted(c_latest["family_name"].dropna().unique().tolist())

module_options = ["Overview", "Pathogens", "Chemicals"]
st.sidebar.radio(
    "Module",
    module_options,
    index=index_for(module_options, st.session_state.get("selected_module", "Overview")),
    key="module_radio",
)
module = st.session_state.get("module_radio", st.session_state.get("selected_module", "Overview"))
st.session_state["selected_module"] = module

if module == "Overview":
    title_col, cta_col = st.columns([4.6, 1.4])
    with title_col:
        st.title("Hotspot Finder")
        st.caption(
            "Final committee-scoped build: Ontario, British Columbia, and Quebec with Pathogens and explicitly named Chemicals."
        )
    with cta_col:
        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
        if st.button("＋ Test Site-Specific Data", use_container_width=True, type="primary"):
            st.session_state["show_upload_options"] = not st.session_state.get("show_upload_options", False)

    if st.session_state.get("show_upload_options", False):
        st.info("Choose which dashboard you want to test with pasted CSV data.")
        u1, u2 = st.columns(2)
        with u1:
            if st.button("Open Pathogens paste tester", use_container_width=True):
                go_to_pathogens(paste=True)
        with u2:
            if st.button("Open Chemicals paste tester", use_container_width=True):
                go_to_chemicals(paste=True)

    st.markdown("### Provincial coverage snapshot")
    cards = st.columns(3)
    for card, prov in zip(cards, CHOSEN_PROVINCES):
        p_subset = p_latest[p_latest["province"] == prov]
        c_subset = c_latest[c_latest["province_scope"] == prov]
        active_sites = int(p_subset["Location"].dropna().nunique())
        top_score = float(p_subset["hotspot_score"].max()) if not p_subset.empty else float("nan")
        chemical_series = int(c_subset[["wwtp_code", "analyte_name", "location_group", "unit"]].drop_duplicates().shape[0])
        with card:
            render_province_card(prov, active_sites, top_score, chemical_series)

    st.markdown("### Chemical families tracked")
    st.markdown(
        "<div class='chip-wrap'>" + "".join([f"<span class='chip'>{fam}</span>" for fam in chemical_families]) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Preview insights")
    top_pathogen = p_latest.sort_values("hotspot_score", ascending=False).iloc[0]
    top_chemical = c_latest.sort_values("hotspot_score", ascending=False).iloc[0]

    a1, a2 = st.columns(2)
    with a1:
        render_alert_card(
            "Top Pathogen Alert",
            f"{top_pathogen['pathogen_name']} — {top_pathogen['Location']}, {top_pathogen['province']}",
            float(top_pathogen["hotspot_score"]),
            f"Current value: {top_pathogen['current_value']:.3f} | Band: {top_pathogen['alert_band']}",
            "Click below to jump directly into the Pathogens module for the deep dive.",
        )
        if st.button("Open Pathogens deep dive", use_container_width=True):
            go_to_pathogens(
                {
                    "province": top_pathogen["province"],
                    "pathogen": top_pathogen["pathogen_name"],
                    "city": top_pathogen.get("city", "All"),
                    "site": top_pathogen["Location"],
                }
            )
    with a2:
        render_alert_card(
            "Top Chemical Alert",
            f"{top_chemical['family_name']} — {top_chemical['analyte_name']} ({top_chemical['wwtp_code']}, {top_chemical['province_scope']})",
            float(top_chemical["hotspot_score"]),
            f"Location: {top_chemical['location_group']} | Band: {top_chemical['alert_band']}",
            "Click below to jump directly into the Chemicals module for the deep dive.",
        )
        if st.button("Open Chemicals deep dive", use_container_width=True):
            go_to_chemicals(
                {
                    "province": top_chemical["province_scope"],
                    "family": top_chemical["family_name"],
                    "location_group": top_chemical["location_group"],
                    "analyte": top_chemical["analyte_name"],
                    "wwtp_code": top_chemical["wwtp_code"],
                    "unit": top_chemical["unit"],
                }
            )

    with st.expander("Methodology, scope rationale, and source notes"):
        st.markdown(
            """
            **Rebuilt scope from committee feedback**

            - **Provinces selected:** Ontario, British Columbia, and Quebec.  
            - **Why these three:** they provide the strongest practical cross-signal subset from the current data. Ontario is the strongest overall, British Columbia is the strongest second, and Quebec is the most usable third province.  
            - **Pathogens:** province scope comes directly from the source dataset.  
            - **Chemicals:** province scope is approximated from WWTP receiving-water context because the processed chemistry files do not carry a clean explicit province field. Great Lakes was used for Ontario, Pacific Ocean for British Columbia, and St. Lawrence River for Quebec.  
            - **Why the app still matters:** raw trend charts remain hard to compare fairly because wastewater interpretation is affected by flow, population served, irregular sampling, non-detects, and data quality.  
            - **Custom testing:** the app includes a paste-your-own-CSV workflow so a user can quickly test site-specific data without replacing the underlying processed files.
            """
        )

elif module == "Pathogens":
    header_col, upload_col = st.columns([4.2, 1.3])
    with header_col:
        st.subheader("Pathogens - Ontario, British Columbia, Quebec")
        st.caption("Interactive pathogen dashboard with visualized scoring logic, cleaned trend charting, and styled latest-record table.")
    with upload_col:
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        render_pathogen_upload_popover()

    if st.session_state.get("upload_target") == "Pathogens":
        st.info("To test site-specific data, use the **Upload Custom CSV** button in the top right of this page.")
        st.session_state["upload_target"] = None
    pref = st.session_state.pop("pathogen_prefill", None)
    f1, f2, f3, f4 = st.columns(4)
    province_options = CHOSEN_PROVINCES
    province = f1.selectbox("Province", province_options, index=index_for(province_options, pref.get("province") if pref else province_options[0]))
    p1 = p_latest[p_latest["province"] == province].copy()

    pathogen_options = sorted(p1["pathogen_name"].dropna().unique().tolist())
    pathogen_default = pref.get("pathogen") if pref else pathogen_options[0]
    pathogen = f2.selectbox("Pathogen", pathogen_options, index=index_for(pathogen_options, pathogen_default))
    p2 = p1[p1["pathogen_name"] == pathogen].copy()

    city_options = ["All"] + sorted(p2["city"].dropna().unique().tolist())
    city_default = pref.get("city") if pref else "All"
    city = f3.selectbox("City", city_options, index=index_for(city_options, city_default))
    if city != "All":
        p2 = p2[p2["city"] == city]

    site_options = sorted(p2["Location"].dropna().unique().tolist())
    site_default = pref.get("site") if pref else site_options[0]
    site = f4.selectbox("Site", site_options, index=index_for(site_options, site_default))

    focus = p2[p2["Location"] == site].sort_values("hotspot_score", ascending=False).head(1)
    if focus.empty:
        st.warning("No rows for this selection.")
    else:
        row = focus.iloc[0]
        render_pathogen_kpi_equation(row)

        hist = p_hist[(p_hist["province"] == province) & (p_hist["pathogen_name"] == pathogen) & (p_hist["Location"] == site)].sort_values("weekstart")
        st.markdown("### Historical trajectory")
        render_pathogen_history_chart(hist)

        st.markdown("### Latest scored records in this filtered set")
        render_pathogen_table(p2)

elif module == "Chemicals":
    header_left, header_mid, header_right = st.columns([4.2, 1.2, 1.6])
    with header_left:
        st.subheader("Chemicals")
    with header_mid:
        with st.popover("ℹ️ Scope Logic", use_container_width=True):
            st.markdown(
                """
                Province scope is assigned from WWTP receiving-water context in the processed public dataset:

                - **Great Lakes → Ontario**
                - **Pacific Ocean → British Columbia**
                - **St. Lawrence River → Quebec**

                This keeps the committee build focused on three provinces with the strongest available coverage.
                """
            )
    with header_right:
        render_chemical_upload_popover()

    if st.session_state.get("upload_target") == "Chemicals":
        st.info("Use the **＋ Upload Custom CSV** button in the top right to test site-specific chemical data.")
        st.session_state["upload_target"] = None

    pref = st.session_state.pop("chemical_prefill", None)
    f1, f2, f3, f4 = st.columns(4)
    province_options = CHOSEN_PROVINCES
    province_default = pref.get("province") if pref else province_options[0]
    province = f1.selectbox("Province", province_options, index=index_for(province_options, province_default))
    c1 = c_latest[c_latest["province_scope"] == province].copy()

    family_options = sorted(c1["family_name"].dropna().unique().tolist())
    family_default = pref.get("family") if pref else family_options[0]
    family = f2.selectbox("Chemical family", family_options, index=index_for(family_options, family_default))
    c2 = c1[c1["family_name"] == family].copy()

    location_options = sorted(c2["location_group"].dropna().unique().tolist())
    location_default = pref.get("location_group") if pref else location_options[0]
    location_group = f3.selectbox("Location group", location_options, index=index_for(location_options, location_default))
    c3 = c2[c2["location_group"] == location_group].copy()

    analyte_options = sorted(c3["analyte_name"].dropna().unique().tolist())
    analyte_default = pref.get("analyte") if pref else analyte_options[0]
    analyte = f4.selectbox("Analyte", analyte_options, index=index_for(analyte_options, analyte_default))
    c4 = c3[c3["analyte_name"] == analyte].copy()

    if pref and pref.get("wwtp_code") in c4["wwtp_code"].tolist():
        series_key = c4[(c4["wwtp_code"] == pref.get("wwtp_code")) & (c4["unit"] == pref.get("unit"))].sort_values("hotspot_score", ascending=False).head(1)
        if series_key.empty:
            series_key = c4.sort_values("hotspot_score", ascending=False).head(1)
    else:
        series_key = c4.sort_values("hotspot_score", ascending=False).head(1)

    if series_key.empty:
        st.warning("No rows for this chemical selection.")
    else:
        row = series_key.iloc[0]
        wwtp = row["wwtp_code"]
        unit = row["unit"]
        hist = c_hist[
            (c_hist["province_scope"] == province)
            & (c_hist["family_name"] == family)
            & (c_hist["location_group"] == location_group)
            & (c_hist["analyte_name"] == analyte)
            & (c_hist["wwtp_code"] == wwtp)
            & (c_hist["unit"] == unit)
        ].sort_values("collection_date")

        render_chemical_kpi_equation(row)
        render_chemical_predictive_chart(hist)
        st.markdown("### Latest scored records in this filtered set")
        render_chemical_table(c4)
