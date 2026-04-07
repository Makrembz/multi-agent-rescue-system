from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rescue_system import (  # noqa: E402
    drone1_node,
    drone2_node,
    get_active_environment,
    initialize_state,
    MissionLogger,
    robot1_node,
    robot2_node,
    show_graph_structure,
    supervisor_node,
    supervisor_router,
)


st.set_page_config(page_title="Multi-Agent Rescue Dashboard", page_icon="🚑", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1480px;
        }
        .panel-card {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 14px 16px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.04), rgba(15, 23, 42, 0.015));
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .agent-title {
            font-size: 1.03rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .muted {
            color: #64748b;
            font-size: 0.92rem;
        }
        .log-item {
            padding: 8px 10px;
            border-radius: 10px;
            margin-bottom: 6px;
            border-left: 5px solid #94a3b8;
            background: rgba(148, 163, 184, 0.08);
            font-size: 0.92rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


AGENT_META = {
    "Supervisor": {"icon": "★", "color": "#7c3aed"},
    "Drone1": {"icon": "▲", "color": "#06b6d4"},
    "Drone2": {"icon": "▲", "color": "#0891b2"},
    "Robot1": {"icon": "●", "color": "#f97316"},
    "Robot2": {"icon": "●", "color": "#fb7185"},
}


def _count_victims(grid: list[list[str]]) -> int:
    return sum(1 for row in grid for cell in row if cell == "victim")


def _bootstrap_session() -> None:
    if "rescue_state" not in st.session_state:
        _reset_simulation()
    if "auto_run" not in st.session_state:
        st.session_state.auto_run = False
    if "last_route" not in st.session_state:
        st.session_state.last_route = "supervisor"
    if "mission_logger" not in st.session_state:
        st.session_state.mission_logger = MissionLogger()
    if "debug_router_robot_seen" not in st.session_state:
        st.session_state.debug_router_robot_seen = False
    if "debug_assigned_nonempty_seen" not in st.session_state:
        st.session_state.debug_assigned_nonempty_seen = False
    if "debug_robot_called_seen" not in st.session_state:
        st.session_state.debug_robot_called_seen = False


def _reset_simulation() -> None:
    st.session_state.rescue_state = initialize_state()
    st.session_state.environment = get_active_environment()
    st.session_state.total_victims = _count_victims(st.session_state.rescue_state["grid"])
    st.session_state.auto_run = False
    st.session_state.last_route = "supervisor"
    st.session_state.last_status = "Simulation reset"
    st.session_state.mission_logger = MissionLogger()
    st.session_state.debug_router_robot_seen = False
    st.session_state.debug_assigned_nonempty_seen = False
    st.session_state.debug_robot_called_seen = False


def _advance_cycle() -> None:
    state = st.session_state.rescue_state
    environment = get_active_environment()

    state = supervisor_node(state)
    router_hint = supervisor_router(state)

    if router_hint != "end":
        # Run both teams in each round so drones keep exploring while robots execute rescues.
        state = drone1_node(state)
        state = drone2_node(state)
        state = robot1_node(state)
        state = robot2_node(state)

    state = supervisor_node(state)
    environment.update_environment()
    st.session_state.mission_logger.log_step(state)

    st.session_state.rescue_state = state
    st.session_state.environment = environment
    st.session_state.last_route = "drone+robot" if router_hint != "end" else "end"
    st.session_state.last_status = f"Supervisor route hint: {router_hint}; executed: {st.session_state.last_route}"
    st.session_state.debug_router_robot_seen = st.session_state.debug_router_robot_seen or (router_hint == "robot")
    st.session_state.debug_assigned_nonempty_seen = st.session_state.debug_assigned_nonempty_seen or bool(state["assigned_tasks"])
    robot_called_recently = any(
        record["agent"] in {"Robot1", "Robot2"} and str(record.get("action", "")) in {"perceive", "decide", "act"}
        for record in state["action_log"][-20:]
    )
    st.session_state.debug_robot_called_seen = st.session_state.debug_robot_called_seen or robot_called_recently

    if state["mission_complete"]:
        st.session_state.auto_run = False


def _energy_color(energy: int | None) -> str:
    if energy is None:
        return "#94a3b8"
    if energy > 50:
        return "#22c55e"
    if energy > 20:
        return "#f59e0b"
    return "#ef4444"


def _agent_status_text(state: dict, agent_id: str) -> str:
    for record in reversed(state["action_log"]):
        if record["agent"] == agent_id:
            return record["action"]
    return "idle"


def _render_grid(state: dict) -> None:
    grid = state["grid"]
    fig, ax = plt.subplots(figsize=(8.6, 8.6))

    cell_map = {
        "empty": 0,
        "safe": 0,
        "obstacle": 1,
        "fire": 2,
        "victim": 3,
        "rescued": 4,
    }
    colors = ["#8fbc8f", "#9ca3af", "#f97316", "#facc15", "#60a5fa"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    numeric_grid = [[cell_map.get(cell, 0) for cell in row] for row in grid]
    ax.imshow(numeric_grid, cmap=cmap, norm=norm)

    ax.set_xticks(range(len(grid)))
    ax.set_yticks(range(len(grid)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([x - 0.5 for x in range(1, len(grid))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(grid))], minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.8, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Live Rescue Grid", fontsize=15, fontweight="bold")

    for agent_id, position in state["agent_positions"].items():
        row, col = position
        if agent_id.startswith("Drone"):
            ax.scatter(col, row, marker="^", s=220, c="#22d3ee", edgecolors="#ffffff", linewidths=1.5, zorder=5)
        elif agent_id.startswith("Robot"):
            ax.scatter(col, row, marker="o", s=190, facecolors="#ffffff", edgecolors="#111827", linewidths=1.6, zorder=5)
        ax.text(col + 0.14, row - 0.18, agent_id.replace("Drone", "D").replace("Robot", "R"), fontsize=8.5, weight="bold")

    legend_items = [
        Patch(facecolor="#9ca3af", label="Obstacle"),
        Patch(facecolor="#f97316", label="Fire"),
        Patch(facecolor="#8fbc8f", label="Empty / Safe"),
        Patch(facecolor="#facc15", label="Victim"),
        Patch(facecolor="#60a5fa", label="Rescued"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)
    ax.set_xlim(-0.5, len(grid) - 0.5)
    ax.set_ylim(len(grid) - 0.5, -0.5)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_agent_card(state: dict, agent_id: str) -> None:
    energy = state["agent_energy"].get(agent_id)
    position = state["agent_positions"].get(agent_id, ("-", "-"))
    current_action = _agent_status_text(state, agent_id)
    icon = AGENT_META[agent_id]["icon"]
    accent = AGENT_META[agent_id]["color"]
    energy_color = _energy_color(energy)
    display_energy = "N/A" if energy is None else f"{energy}%"

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="agent-title" style="color:{accent};">{icon} {agent_id}</div>
            <div class="muted">Position: {position}</div>
            <div class="muted">Status: {current_action}</div>
            <div style="margin-top: 10px; font-size: 0.88rem; font-weight: 700;">Energy: {display_energy}</div>
            <div style="margin-top: 4px; width: 100%; background: #e2e8f0; border-radius: 999px; overflow: hidden; height: 12px;">
                <div style="width: {0 if energy is None else energy}%; height: 100%; background: {energy_color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_panel(state: dict) -> None:
    for agent_id in ["Supervisor", "Drone1", "Drone2", "Robot1", "Robot2"]:
        _render_agent_card(state, agent_id)


def _render_metrics(state: dict) -> None:
    total_victims = st.session_state.total_victims or 1
    detected = len(state["detected_victims"])
    rescued = len(state["rescued_victims"])
    current_step = state["mission_step"]
    progress = min(1.0, rescued / total_victims)

    col1, col2 = st.columns(2)
    col1.metric("Victims Detected", f"{detected}/{total_victims}")
    col2.metric("Victims Rescued", f"{rescued}/{total_victims}")
    st.metric("Current Step", current_step)
    st.progress(progress, text=f"Mission Progress: {int(progress * 100)}%")


def _render_assignments(state: dict) -> None:
    rows = []
    for robot_id in ["Robot1", "Robot2"]:
        victim = state["assigned_tasks"].get(robot_id)
        if victim is None:
            status = "Available" if state["agent_energy"].get(robot_id, 0) > 10 else "Low energy"
            victim_display = "-"
        elif victim in state["rescued_victims"]:
            status = "Rescued"
            victim_display = str(victim)
        else:
            status = "Assigned"
            victim_display = str(victim)
        rows.append({"Robot": robot_id, "Assigned Victim": victim_display, "Status": status})

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_action_log(state: dict) -> None:
    logs = state["action_log"][-10:][::-1]
    if not logs:
        st.info("No actions recorded yet.")
        return

    for record in logs:
        agent = record["agent"]
        meta = AGENT_META.get(agent, {"color": "#64748b", "icon": "•"})
        st.markdown(
            f"""
            <div class="log-item" style="border-left-color: {meta['color']};">
                <strong style="color: {meta['color']};">{meta['icon']} {agent}</strong>
                <span style="color:#334155;"> - {record['action']}</span>
                <span style="float:right; color:#64748b;">step {record['step']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_debug_panel(state: dict) -> None:
    st.write("1) Is supervisor_router ever returning 'robot'?", st.session_state.debug_router_robot_seen)
    st.write("2) Is assigned_tasks ever non-empty?", st.session_state.debug_assigned_nonempty_seen)
    st.write("3) Is robot_node ever being called?", st.session_state.debug_robot_called_seen)
    st.write("Current route", st.session_state.get("last_route", "-"))
    st.write("Current assigned_tasks", state.get("assigned_tasks", {}))


def _coverage_percentage(state: dict) -> float:
    grid = state.get("grid", [])
    if not grid:
        return 0.0

    observed_cells = 0
    for local_map in state.get("local_maps", {}).values():
        for row in local_map:
            observed_cells += sum(1 for cell in row if cell is not None)

    return min(100.0, (observed_cells / (len(grid) * len(grid))) * 100)


def _render_energy_chart(logger: MissionLogger) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = {"Drone1": "#06b6d4", "Drone2": "#0891b2", "Robot1": "#f97316", "Robot2": "#fb7185"}

    for agent_id in ["Drone1", "Drone2", "Robot1", "Robot2"]:
        series = logger.agent_energy_history.get(agent_id, [])
        if not series:
            continue
        ax.plot(range(1, len(series) + 1), series, label=agent_id, linewidth=2.2, color=colors[agent_id])

    ax.set_xlabel("Steps")
    ax.set_ylabel("Energy %")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_timeline_chart(logger: MissionLogger) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    victims = sorted(set(logger.victims_detected_per_step) | set(logger.victims_rescued_per_step))

    if not victims:
        ax.text(0.5, 0.5, "No victim timeline yet", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return

    for index, victim_position in enumerate(victims):
        detect_step = logger.victims_detected_per_step.get(victim_position, 0)
        rescue_step = logger.victims_rescued_per_step.get(victim_position, logger.total_steps)
        duration = max(rescue_step - detect_step, 0.1)
        ax.barh(index, duration, left=detect_step, color="#60a5fa", alpha=0.8)
        ax.barh(index, 0.3, left=detect_step, color="#facc15", alpha=1.0)
        ax.text(rescue_step + 0.12, index, str(victim_position), va="center", fontsize=9)

    ax.set_yticks(range(len(victims)))
    ax.set_yticklabels([str(position) for position in victims])
    ax.set_xlabel("Step")
    ax.set_title("Detection → Rescue Timeline")
    ax.grid(axis="x", alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_report_tab(state: dict) -> None:
    logger = st.session_state.mission_logger
    total_victims = st.session_state.total_victims or 1
    rescued_count = len(state["rescued_victims"])
    success_rate = (rescued_count / total_victims) * 100 if total_victims else 0.0

    st.markdown(
        f"""
        <div class="panel-card" style="margin-bottom: 16px; padding: 18px;">
            <div style="font-size: 1.55rem; font-weight: 800; color: {'#16a34a' if state['mission_complete'] else '#dc2626'};">
                {'MISSION COMPLETE' if state['mission_complete'] else 'IN PROGRESS'}
            </div>
            <div class="muted" style="margin-top: 4px;">Final rescue rate: {rescued_count}/{total_victims} ({success_rate:.0f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Steps", logger.total_steps)
    metric_cols[1].metric("Victims Rescued", f"{rescued_count}/{total_victims}")
    metric_cols[2].metric("Success Rate", f"{success_rate:.0f}%")
    metric_cols[3].metric("Avg Rescue Time", f"{logger.get_avg_rescue_time():.1f} steps")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Energy Chart")
        _render_energy_chart(logger)
    with chart_col2:
        st.subheader("Timeline Chart")
        _render_timeline_chart(logger)

    st.subheader("Metrics Table")
    metrics_table = [
        {
            "Messages Sent": logger.messages_sent,
            "Tasks Assigned": logger.tasks_assigned,
            "Path Replanning": logger.path_replanning_count,
            "Coverage %": f"{_coverage_percentage(state):.1f}%",
        }
    ]
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    st.subheader("Action Log Export")
    export_path = logger.export_json("mission_log.json")
    st.download_button(
        label="Export mission_log.json",
        data=json.dumps(logger.to_dict(), indent=2),
        file_name=export_path.name,
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    _bootstrap_session()

    st.title("Multi-Agent Rescue System")
    st.caption("Streamlit live dashboard for the LangGraph rescue simulation.")

    live_tab, report_tab = st.tabs(["Live Simulation", "Mission Report"])

    with live_tab:
        top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.0])
        with top_left:
            if st.button("Next Step", use_container_width=True):
                _advance_cycle()
        with top_mid:
            if st.button("Run Auto", use_container_width=True):
                st.session_state.auto_run = True
        with top_right:
            if st.button("Reset", use_container_width=True):
                _reset_simulation()
                st.rerun()

        auto_col1, auto_col2 = st.columns([1.0, 5.0])
        with auto_col1:
            if st.button("Pause Auto", use_container_width=True):
                st.session_state.auto_run = False
        with auto_col2:
            st.markdown(f"**{st.session_state.get('last_status', 'Ready')}**")

        if st.session_state.auto_run and not st.session_state.rescue_state["mission_complete"]:
            _advance_cycle()
            time.sleep(0.5)
            st.rerun()

        state = st.session_state.rescue_state

        left_col, right_col = st.columns([2.15, 1.0], gap="large")

        with left_col:
            _render_grid(state)

        with right_col:
            with st.container(border=True):
                st.subheader("Agent Status")
                _render_status_panel(state)

            with st.container(border=True):
                st.subheader("Mission Metrics")
                _render_metrics(state)

            with st.container(border=True):
                st.subheader("Task Assignments")
                _render_assignments(state)

        with st.container(border=True):
            st.subheader("Live Action Log")
            _render_action_log(state)

        with st.expander("Compiled Graph Structure"):
            st.code(show_graph_structure(), language="text")

        with st.expander("Debug Checks"):
            _render_debug_panel(state)

    with report_tab:
        _render_report_tab(st.session_state.rescue_state)


if __name__ == "__main__":
    main()
