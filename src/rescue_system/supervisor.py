from __future__ import annotations

import ast
import re
from typing import Any

from .state import RescueState


def supervisor_node(state: RescueState) -> RescueState:
    """Read global state, allocate victims to robots, and update mission control."""

    _ensure_state_defaults(state)
    _ingest_drone_messages(state)
    _cleanup_stale_assignments(state)

    detected_victims = state["detected_victims"]
    assigned_tasks = state["assigned_tasks"]

    available_robots = _available_robots(state)
    used_robots = set(assigned_tasks.keys())
    assigned_victims = set(assigned_tasks.values())

    for victim_position in detected_victims:
        if victim_position in assigned_victims or victim_position in state["rescued_victims"]:
            continue

        candidate_robot = _closest_available_robot(state, victim_position, available_robots, used_robots)
        if candidate_robot is None:
            _log_action(
                state,
                "Supervisor",
                f"no_robot_available_for_{victim_position}",
                state["mission_step"],
            )
            continue

        assigned_tasks[candidate_robot] = victim_position
        used_robots.add(candidate_robot)
        assigned_victims.add(victim_position)
        print(f"[SUPERVISOR] assigned {candidate_robot} -> {victim_position}")
        _log_action(
            state,
            "Supervisor",
            f"assign_{candidate_robot}_to_{victim_position}",
            state["mission_step"],
        )

    if _all_victims_rescued(state):
        state["mission_complete"] = True
        _log_action(state, "Supervisor", "mission_complete", state["mission_step"])
    else:
        state["mission_complete"] = False

    print(f"[SUPERVISOR] assigned_tasks={dict(state['assigned_tasks'])}")
    _log_action(state, "Supervisor", f"assigned_tasks_{dict(state['assigned_tasks'])}", state["mission_step"])
    return state


def supervisor_router(state: RescueState) -> str:
    """Route the graph to drones, robots, or termination based on shared state."""

    _ensure_state_defaults(state)

    if state["mission_complete"]:
        print("[ROUTER] route=end (mission complete)")
        return "end"

    # Prioritize robot execution once there are active assignments.
    if _robots_can_act(state):
        print(f"[ROUTER] route=robot assigned_tasks={dict(state['assigned_tasks'])}")
        return "robot"

    if _has_unexplored_area(state):
        print("[ROUTER] route=drone (unexplored area remains)")
        return "drone"

    if state["detected_victims"] and not _all_victims_rescued(state):
        print("[ROUTER] route=robot (detected victims pending)")
        return "robot"

    print("[ROUTER] route=drone (default)")
    return "drone"


def _ensure_state_defaults(state: RescueState) -> None:
    if "agent_positions" not in state:
        state["agent_positions"] = {}
    if "agent_energy" not in state:
        state["agent_energy"] = {}
    if "local_maps" not in state:
        state["local_maps"] = {}
    if "detected_victims" not in state:
        state["detected_victims"] = []
    if "assigned_tasks" not in state:
        state["assigned_tasks"] = {}
    if "rescued_victims" not in state:
        state["rescued_victims"] = []
    if "messages" not in state:
        state["messages"] = []
    if "action_log" not in state:
        state["action_log"] = []
    if "mission_step" not in state:
        state["mission_step"] = 0
    if "mission_complete" not in state:
        state["mission_complete"] = False


def _ingest_drone_messages(state: RescueState) -> None:
    for message in state["messages"]:
        if message.get("type") not in {"victim_detected", "victim_confirmed"}:
            continue

        position = _extract_position(message.get("content", ""))
        if position is not None and position not in state["detected_victims"]:
            state["detected_victims"].append(position)
            _log_action(state, "Supervisor", f"ingest_victim_alert_{position}", state["mission_step"])


def _extract_position(content: str) -> tuple[int, int] | None:
    match = re.search(r"\((-?\d+),\s*(-?\d+)\)", content)
    if match is not None:
        return int(match.group(1)), int(match.group(2))

    try:
        parsed = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return None

    if isinstance(parsed, tuple) and len(parsed) == 2 and all(isinstance(value, int) for value in parsed):
        return parsed
    return None


def _cleanup_stale_assignments(state: RescueState) -> None:
    rescued = set(state["rescued_victims"])
    to_remove = [robot_id for robot_id, victim in state["assigned_tasks"].items() if victim in rescued]
    for robot_id in to_remove:
        del state["assigned_tasks"][robot_id]


def _available_robots(state: RescueState) -> list[str]:
    assigned_robots = set(state["assigned_tasks"].keys())
    return [
        agent_id
        for agent_id, energy in state["agent_energy"].items()
        if agent_id.lower().startswith("robot") and agent_id not in assigned_robots and energy > 10
    ]


def _closest_available_robot(
    state: RescueState,
    victim_position: tuple[int, int],
    available_robots: list[str],
    used_robots: set[str],
) -> str | None:
    best_robot: str | None = None
    best_distance: int | None = None

    for robot_id in available_robots:
        if robot_id in used_robots:
            continue

        robot_position = state["agent_positions"].get(robot_id)
        if robot_position is None:
            continue

        distance = _manhattan_distance(robot_position, victim_position)
        if best_distance is None or distance < best_distance or (distance == best_distance and robot_id < best_robot):
            best_robot = robot_id
            best_distance = distance

    return best_robot


def _all_victims_rescued(state: RescueState) -> bool:
    remaining_victims = sum(1 for row in state.get("grid", []) for cell in row if cell == "victim")
    return remaining_victims == 0 and not state["assigned_tasks"]


def _has_unexplored_area(state: RescueState) -> bool:
    for local_map in state["local_maps"].values():
        for row in local_map:
            if any(cell is None for cell in row):
                return True
    return not state["local_maps"]


def _robots_can_act(state: RescueState) -> bool:
    for robot_id in state["assigned_tasks"]:
        if state["agent_energy"].get(robot_id, 0) > 10:
            return True
    return False


def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _log_action(state: RescueState, agent: str, action: str, step: int) -> None:
    state["action_log"].append({"agent": agent, "action": action, "step": step})
