"""
Streamlit dashboard for pipeline visibility and weekly KPI review.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import JobSearchStatus
from core.database import PipelineEntryORM, get_db, init_db


def load_pipeline_rows() -> pd.DataFrame:
    """Load pipeline rows from the database into a DataFrame."""
    init_db()
    with get_db() as db:
        rows = db.query(PipelineEntryORM).all()

    payload: List[Dict[str, object]] = []
    for row in rows:
        payload.append(
            {
                "company_name": row.company_name,
                "role_title": row.role_title,
                "status": row.status,
                "last_activity_at": row.last_activity_at,
                "hot_score": row.hot_score or 0.0,
            }
        )
    return pd.DataFrame(payload)


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute dashboard KPIs from a pipeline dataframe."""
    if df.empty:
        return {
            "total_in_pipeline": 0,
            "active": 0,
            "response_rate": 0.0,
            "interview_rate": 0.0,
            "ghosted_rate": 0.0,
        }

    stage_counts = df["status"].value_counts().to_dict()
    total_applied = (
        stage_counts.get(JobSearchStatus.APPLIED.value, 0)
        + stage_counts.get(JobSearchStatus.RESPONDED.value, 0)
        + stage_counts.get(JobSearchStatus.INTERVIEWING.value, 0)
        + stage_counts.get(JobSearchStatus.OFFER.value, 0)
        + stage_counts.get(JobSearchStatus.REJECTED.value, 0)
        + stage_counts.get(JobSearchStatus.GHOSTED.value, 0)
    )
    total_responded = (
        stage_counts.get(JobSearchStatus.RESPONDED.value, 0)
        + stage_counts.get(JobSearchStatus.INTERVIEWING.value, 0)
        + stage_counts.get(JobSearchStatus.OFFER.value, 0)
    )
    total_interview = (
        stage_counts.get(JobSearchStatus.INTERVIEWING.value, 0)
        + stage_counts.get(JobSearchStatus.OFFER.value, 0)
    )
    active = (
        stage_counts.get(JobSearchStatus.IDENTIFIED.value, 0)
        + stage_counts.get(JobSearchStatus.APPLIED.value, 0)
        + stage_counts.get(JobSearchStatus.RESPONDED.value, 0)
        + stage_counts.get(JobSearchStatus.INTERVIEWING.value, 0)
    )

    return {
        "total_in_pipeline": int(df.shape[0]),
        "active": float(active),
        "response_rate": float(total_responded / max(total_applied, 1)),
        "interview_rate": float(total_interview / max(total_responded, 1)),
        "ghosted_rate": float(stage_counts.get(JobSearchStatus.GHOSTED.value, 0) / max(total_applied, 1)),
    }


def render_dashboard() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Job Search KPI Dashboard", layout="wide")
    st.title("Job Search KPI Dashboard")

    df = load_pipeline_rows()
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("In Pipeline", int(kpis["total_in_pipeline"]))
    c2.metric("Active Opportunities", int(kpis["active"]))
    c3.metric("Response Rate", f"{kpis['response_rate']:.0%}")
    c4.metric("Interview Rate", f"{kpis['interview_rate']:.0%}")

    if df.empty:
        st.info("No pipeline data yet. Run the orchestrator cycle first.")
        return

    stage_order = [status.value for status in JobSearchStatus]
    status_counts = (
        df["status"]
        .value_counts()
        .reindex(stage_order, fill_value=0)
        .reset_index()
        .rename(columns={"index": "stage", "status": "count"})
    )
    funnel_fig = px.funnel(status_counts, x="count", y="stage", title="Pipeline Funnel")
    st.plotly_chart(funnel_fig, use_container_width=True)

    cutoff = datetime.utcnow() - timedelta(days=7)
    stale_df = df[df["last_activity_at"] < cutoff]
    st.subheader("Follow-up Alerts (No activity in 7+ days)")
    if stale_df.empty:
        st.success("No stale entries found.")
    else:
        st.dataframe(
            stale_df[["company_name", "role_title", "status", "last_activity_at", "hot_score"]]
            .sort_values(by="last_activity_at", ascending=True),
            use_container_width=True,
        )

    st.subheader("Top Bottleneck Signals")
    ghosted_rate = kpis["ghosted_rate"]
    if ghosted_rate > 0.4:
        st.warning("High ghosted rate detected. Increase structured follow-up cadence.")
    elif kpis["response_rate"] < 0.2:
        st.warning("Low response rate detected. Revisit profile-to-role match and outreach quality.")
    else:
        st.info("No severe bottlenecks detected this cycle.")


if __name__ == "__main__":
    render_dashboard()

