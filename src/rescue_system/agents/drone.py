from __future__ import annotations

from typing import Any

from ..environment import Environment
from ..knowledge_base import KnowledgeBase
from ..state import RescueState

ACTIVE_ENVIRONMENT: Environment | None = None
KNOWLEDGE_BASES: dict[str, KnowledgeBase] = {}


def register_environment(environment: Environment) -> None:
    """Register the shared environment used by all drone nodes."""

    global ACTIVE_ENVIRONMENT
    ACTIVE_ENVIRONMENT = environment


def get_active_environment() -> Environment:
    """Return the currently registered shared environment."""

    if ACTIVE_ENVIRONMENT is None:
        raise RuntimeError("No active environment registered. Call register_environment() first.")
    return ACTIVE_ENVIRONMENT


def drone_node(state: RescueState) -> RescueState:
    """Default drone node mapped to Drone1 for LangGraph compatibility."""

    return _run_drone_cycle(state, "Drone1")


def drone1_node(state: RescueState) -> RescueState:
    return _run_drone_cycle(state, "Drone1")


def drone2_node(state: RescueState) -> RescueState:
    return _run_drone_cycle(state, "Drone2")


def _run_drone_cycle(state: RescueState, agent_id: str) -> RescueState:
    environment = _require_environment()
    knowledge_base = KNOWLEDGE_BASES.setdefault(agent_id, KnowledgeBase())
    _ensure_state_defaults(state, agent_id, environment)

    position = state["agent_positions"][agent_id]
    energy = state["agent_energy"].get(agent_id, 100)
    radius = 3 if agent_id == "Drone1" else 2

    _log_action(state, agent_id, "perceive", state["mission_step"])
    observation = environment.get_local_observation(position, radius=radius, noise=True)

    observation_payload, confirmed_victims = _build_observation_payload(
        environment,
        position,
        observation,
        state,
        agent_id,
    )
    knowledge_base.update_belief_state(observation_payload)
    conclusions = knowledge_base.infer()

    for victim_position in confirmed_victims:
        if victim_position not in state["detected_victims"]:
            state["detected_victims"].append(victim_position)
        _send_message(
            state,
            sender=agent_id,
            receiver="Supervisor",
            message_type="victim_detected",
            content=f"Confirmed victim at {victim_position}",
        )

    if energy < 10:
        _log_action(state, agent_id, "stop_low_energy", state["mission_step"])
        state["agent_energy"][agent_id] = energy
        _merge_local_map(state, agent_id, position, observation)
        return state

    next_position = _decide_next_position(
        environment=environment,
        state=state,
        agent_id=agent_id,
        position=position,
        conclusions=conclusions,
    )

    _log_action(state, agent_id, "reason", state["mission_step"])
    _log_action(state, agent_id, "decide", state["mission_step"])

    if next_position != position and environment.is_valid_move(next_position):
        state["agent_positions"][agent_id] = next_position
        state["agent_energy"][agent_id] = max(0, energy - 2)
        _log_action(state, agent_id, f"move_to_{next_position}", state["mission_step"])
    else:
        state["agent_energy"][agent_id] = energy
        _log_action(state, agent_id, "hold_position", state["mission_step"])

    new_position = state["agent_positions"][agent_id]
    _merge_local_map(state, agent_id, new_position, observation)
    _update_detection_from_position(environment, state, agent_id, new_position)
    _send_message(
        state,
        sender=agent_id,
        receiver=_teammate_id(agent_id),
        message_type="coverage_update",
        content=f"position={new_position}",
    )
    _log_action(state, agent_id, "act", state["mission_step"])

    state["mission_step"] += 1

    return state


def _require_environment() -> Environment:
    if ACTIVE_ENVIRONMENT is None:
        raise RuntimeError("No active environment registered. Call register_environment() first.")
    return ACTIVE_ENVIRONMENT


def _ensure_state_defaults(state: RescueState, agent_id: str, environment: Environment) -> None:
    if "agent_positions" not in state:
        state["agent_positions"] = {}
    if "agent_energy" not in state:
        state["agent_energy"] = {}
    if "local_maps" not in state:
        state["local_maps"] = {}
    if "detected_victims" not in state:
        state["detected_victims"] = []
    if "messages" not in state:
        state["messages"] = []
    if "action_log" not in state:
        state["action_log"] = []
    if "mission_step" not in state:
        state["mission_step"] = 0
    if "mission_complete" not in state:
        state["mission_complete"] = False

    if agent_id not in state["agent_positions"]:
        state["agent_positions"][agent_id] = _find_spawn_position(environment, agent_id)
    if agent_id not in state["agent_energy"]:
        state["agent_energy"][agent_id] = 100

    if agent_id not in state["local_maps"]:
        state["local_maps"][agent_id] = [[None for _ in range(environment.n)] for _ in range(environment.n)]


def _find_spawn_position(environment: Environment, agent_id: str) -> tuple[int, int]:
    preferred_columns = {"Drone1": 0, "Drone2": 1}
    preferred_position = (0, preferred_columns.get(agent_id, 0))
    if environment.is_valid_move(preferred_position):
        return preferred_position

    for row in range(environment.n):
        for col in range(environment.n):
            candidate = (row, col)
            if environment.is_valid_move(candidate):
                return candidate

    return (0, 0)


def _build_observation_payload(
    environment: Environment,
    agent_position: tuple[int, int],
    observation: list[list[Any]],
    state: RescueState,
    agent_id: str,
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    radius = len(observation) // 2
    cell_states: dict[tuple[int, int], str] = {}
    victim_positions: list[tuple[int, int]] = []
    danger_zones: list[tuple[int, int]] = []
    explored_cells: list[tuple[int, int]] = []
    confirmed_victims: list[tuple[int, int]] = []

    for row_offset, row in enumerate(observation):
        for col_offset, cell in enumerate(row):
            if cell is None:
                continue

            row_position = agent_position[0] + row_offset - radius
            col_position = agent_position[1] + col_offset - radius
            absolute_position = (row_position, col_position)

            cell_states[absolute_position] = cell
            explored_cells.append(absolute_position)

            if cell == "victim":
                victim_positions.append(absolute_position)
                if environment.get_cell(absolute_position) == "victim":
                    confirmed_victims.append(absolute_position)
            if cell in {"fire", "obstacle"}:
                danger_zones.append(absolute_position)

    payload: dict[str, Any] = {
        "cell_states": cell_states,
        "victim_positions": victim_positions,
        "danger_zones": danger_zones,
        "agent_position": agent_position,
        "energy": state["agent_energy"].get(agent_id, 100),
        "explored_cells": explored_cells,
        "assigned_victims": list(state.get("assigned_tasks", {}).values()),
    }

    return payload, confirmed_victims


def _decide_next_position(
    environment: Environment,
    state: RescueState,
    agent_id: str,
    position: tuple[int, int],
    conclusions: list[dict[str, Any]],
) -> tuple[int, int]:
    teammate_position = state.get("agent_positions", {}).get(_teammate_id(agent_id))

    if any(conclusion["rule"] == "low_energy_return_to_base" for conclusion in conclusions):
        return _step_toward(environment, position, environment.safe_position, prefer_unexplored=False) or position

    snake_step = _snake_next_position(environment, state, agent_id, position, teammate_position)
    if snake_step is not None:
        return snake_step

    return _step_toward(
        environment,
        position,
        _first_unexplored_target(environment, state, agent_id, teammate_position),
        prefer_unexplored=True,
    ) or position


def _snake_next_position(
    environment: Environment,
    state: RescueState,
    agent_id: str,
    position: tuple[int, int],
    teammate_position: tuple[int, int] | None,
) -> tuple[int, int] | None:
    snake_order = _snake_order(environment.n, agent_id)
    current_index = snake_order.index(position) if position in snake_order else -1

    def _one_step_toward_target(target: tuple[int, int]) -> tuple[int, int] | None:
        # Movement must be local: exactly one neighboring square per step.
        neighbors = environment._neighbors(position)
        preferred_neighbors = [neighbor for neighbor in neighbors if environment.is_valid_move(neighbor)]

        non_overlap_neighbors = [neighbor for neighbor in preferred_neighbors if not _too_close(neighbor, teammate_position)]
        candidate_pool = non_overlap_neighbors or preferred_neighbors
        if not candidate_pool:
            return None

        return min(candidate_pool, key=lambda cell: _manhattan_distance(cell, target))

    for candidate in snake_order[current_index + 1 :]:
        step = _one_step_toward_target(candidate)
        if step is not None:
            return step

    for candidate in snake_order[: max(current_index, 0)]:
        step = _one_step_toward_target(candidate)
        if step is not None:
            return step

    return None


def _first_unexplored_target(
    environment: Environment,
    state: RescueState,
    agent_id: str,
    teammate_position: tuple[int, int] | None,
) -> tuple[int, int]:
    local_map = state["local_maps"].get(agent_id, [])

    preferred_rows = [row for row in range(environment.n) if _is_preferred_row(agent_id, row)]
    secondary_rows = [row for row in range(environment.n) if row not in preferred_rows]

    for row in preferred_rows + secondary_rows:
        columns = range(environment.n) if row % 2 == 0 else range(environment.n - 1, -1, -1)
        for col in columns:
            if (not local_map or local_map[row][col] is None) and not _too_close((row, col), teammate_position):
                return (row, col)

    for row in range(environment.n):
        for col in range(environment.n):
            if not local_map or local_map[row][col] is None:
                return (row, col)
    return environment.safe_position


def _snake_order(size: int, agent_id: str) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    preferred_rows = [row for row in range(size) if _is_preferred_row(agent_id, row)]
    secondary_rows = [row for row in range(size) if row not in preferred_rows]

    for row in preferred_rows + secondary_rows:
        columns = range(size) if row % 2 == 0 else range(size - 1, -1, -1)
        for col in columns:
            order.append((row, col))
    return order


def _is_preferred_row(agent_id: str, row: int) -> bool:
    # Drone1 favors even rows, Drone2 favors odd rows.
    return (row % 2 == 0) if agent_id == "Drone1" else (row % 2 == 1)


def _too_close(position: tuple[int, int], teammate_position: tuple[int, int] | None) -> bool:
    if teammate_position is None:
        return False
    return abs(position[0] - teammate_position[0]) + abs(position[1] - teammate_position[1]) <= 1


def _teammate_id(agent_id: str) -> str:
    return "Drone2" if agent_id == "Drone1" else "Drone1"


def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(
    environment: Environment,
    current: tuple[int, int],
    target: tuple[int, int] | None,
    prefer_unexplored: bool,
) -> tuple[int, int] | None:
    if target is None:
        return None

    row, col = current
    target_row, target_col = target
    candidates = []

    if target_row > row:
        candidates.append((row + 1, col))
    elif target_row < row:
        candidates.append((row - 1, col))

    if target_col > col:
        candidates.append((row, col + 1))
    elif target_col < col:
        candidates.append((row, col - 1))

    for candidate in candidates:
        if not environment._in_bounds(candidate):
            continue
        if environment.is_valid_move(candidate):
            return candidate

    if prefer_unexplored:
        for candidate in environment._neighbors(current):
            if environment.is_valid_move(candidate):
                return candidate

    return None


def _merge_local_map(
    state: RescueState,
    agent_id: str,
    position: tuple[int, int],
    observation: list[list[Any]],
) -> None:
    local_map = state["local_maps"][agent_id]
    radius = len(observation) // 2

    for row_offset, row in enumerate(observation):
        for col_offset, cell in enumerate(row):
            if cell is None:
                continue
            absolute_row = position[0] + row_offset - radius
            absolute_col = position[1] + col_offset - radius
            if 0 <= absolute_row < len(local_map) and 0 <= absolute_col < len(local_map[absolute_row]):
                local_map[absolute_row][absolute_col] = cell


def _update_detection_from_position(
    environment: Environment,
    state: RescueState,
    agent_id: str,
    position: tuple[int, int],
) -> None:
    if environment.get_cell(position) == "victim" and position not in state["detected_victims"]:
        state["detected_victims"].append(position)
        _send_message(
            state,
            sender=agent_id,
            receiver="Supervisor",
            message_type="victim_confirmed",
            content=f"Victim confirmed at {position}",
        )


def _send_message(
    state: RescueState,
    sender: str,
    receiver: str,
    message_type: str,
    content: str,
) -> None:
    state["messages"].append(
        {
            "sender": sender,
            "receiver": receiver,
            "type": message_type,
            "content": content,
        }
    )


def _log_action(state: RescueState, agent: str, action: str, step: int) -> None:
    state["action_log"].append({"agent": agent, "action": action, "step": step})
