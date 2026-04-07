from __future__ import annotations

import heapq
from typing import Any

from ..environment import Environment
from ..knowledge_base import KnowledgeBase
from ..state import RescueState
from . import drone as drone_runtime

ROBOT_KNOWLEDGE_BASES: dict[str, KnowledgeBase] = {}


def robot_node(state: RescueState) -> RescueState:
    """Default robot node mapped to Robot1 for LangGraph compatibility."""

    return _run_robot_cycle(state, "Robot1")


def robot1_node(state: RescueState) -> RescueState:
    return _run_robot_cycle(state, "Robot1")


def robot2_node(state: RescueState) -> RescueState:
    return _run_robot_cycle(state, "Robot2")


def _run_robot_cycle(state: RescueState, agent_id: str) -> RescueState:
    environment = _require_environment()
    knowledge_base = ROBOT_KNOWLEDGE_BASES.setdefault(agent_id, KnowledgeBase())
    _ensure_state_defaults(state, agent_id, environment)

    position = state["agent_positions"][agent_id]
    assigned_target = state["assigned_tasks"].get(agent_id)
    energy = state["agent_energy"].get(agent_id, 100)
    print(f"[ROBOT] {agent_id} called. assigned_target={assigned_target}, position={position}, energy={energy}")

    _log_action(state, agent_id, "perceive", state["mission_step"])

    if assigned_target is None:
        knowledge_base.update_belief_state(
            {
                "agent_position": position,
                "energy": energy,
                "explored_cells": _known_explored_cells(state, agent_id),
            }
        )
        knowledge_base.infer()
        _log_action(state, agent_id, "idle_no_task", state["mission_step"])
        state["mission_step"] += 1
        return state

    local_map = state["local_maps"].get(agent_id, [])
    belief_observation = _build_belief_from_local_map(state, agent_id, position, local_map)
    knowledge_base.update_belief_state(belief_observation)
    conclusions = knowledge_base.infer()

    _log_action(state, agent_id, "reason", state["mission_step"])

    if energy < 10:
        _log_action(state, agent_id, "stop_low_energy", state["mission_step"])
        state["agent_energy"][agent_id] = energy
        state["mission_step"] += 1
        return state

    if _is_adjacent_to_victim(environment, position, assigned_target):
        _perform_rescue(state, agent_id, assigned_target)
        _log_action(state, agent_id, "rescue", state["mission_step"])
        state["mission_step"] += 1
        return state

    goal_cells = _approach_cells(environment, assigned_target)
    if not goal_cells:
        _log_action(state, agent_id, "no_safe_approach", state["mission_step"])
        fallback_path = _fallback_path_toward_target(environment, position, assigned_target)
        if len(fallback_path) > 1:
            state["agent_positions"][agent_id] = fallback_path[1]
            state["agent_energy"][agent_id] = max(0, energy - 3)
            _log_action(state, agent_id, f"fallback_move_to_{fallback_path[1]}", state["mission_step"])
        else:
            _log_action(state, agent_id, "hold_no_reachable_cell", state["mission_step"])
        state["mission_step"] += 1
        return state

    path = _plan_a_star_path(environment, knowledge_base, position, goal_cells)
    if len(path) > 1 and _path_contains_known_danger(knowledge_base, path):
        _log_action(state, agent_id, "replan_due_to_danger", state["mission_step"])
        path = _plan_a_star_path(environment, knowledge_base, position, goal_cells, expand_danger_buffer=True)

    _log_action(state, agent_id, "decide", state["mission_step"])

    if len(path) <= 1:
        _log_action(state, agent_id, "no_path_found", state["mission_step"])
        fallback_path = _fallback_path_toward_target(environment, position, assigned_target)
        if len(fallback_path) > 1:
            path = fallback_path
            _log_action(state, agent_id, f"fallback_path_{fallback_path[-1]}", state["mission_step"])

    if len(path) > 1:
        next_position = path[1]
        if environment.is_valid_move(next_position):
            state["agent_positions"][agent_id] = next_position
            state["agent_energy"][agent_id] = max(0, energy - 3)
            _log_action(state, agent_id, f"move_to_{next_position}", state["mission_step"])
        else:
            _log_action(state, agent_id, "blocked_move", state["mission_step"])
    else:
        _log_action(state, agent_id, "hold_position", state["mission_step"])

    new_position = state["agent_positions"][agent_id]
    if _is_adjacent_to_victim(environment, new_position, assigned_target):
        _perform_rescue(state, agent_id, assigned_target)
        _log_action(state, agent_id, "rescue", state["mission_step"])

    _log_action(state, agent_id, "act", state["mission_step"])
    state["mission_step"] += 1
    return state


def _require_environment() -> Environment:
    if drone_runtime.ACTIVE_ENVIRONMENT is None:
        raise RuntimeError("No active environment registered. Call register_environment() first.")
    return drone_runtime.ACTIVE_ENVIRONMENT


def _ensure_state_defaults(state: RescueState, agent_id: str, environment: Environment) -> None:
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

    if agent_id not in state["agent_positions"]:
        state["agent_positions"][agent_id] = _find_spawn_position(environment, agent_id)
    if agent_id not in state["agent_energy"]:
        state["agent_energy"][agent_id] = 100
    if agent_id not in state["local_maps"]:
        state["local_maps"][agent_id] = [[None for _ in range(environment.n)] for _ in range(environment.n)]


def _find_spawn_position(environment: Environment, agent_id: str) -> tuple[int, int]:
    preferred_columns = {"Robot1": environment.n - 1, "Robot2": max(environment.n - 2, 0)}
    preferred_position = (environment.n - 1, preferred_columns.get(agent_id, environment.n - 1))
    if environment.is_valid_move(preferred_position):
        return preferred_position

    for row in range(environment.n - 1, -1, -1):
        for col in range(environment.n - 1, -1, -1):
            candidate = (row, col)
            if environment.is_valid_move(candidate):
                return candidate

    return (environment.n - 1, environment.n - 1)


def _build_belief_from_local_map(
    state: RescueState,
    agent_id: str,
    position: tuple[int, int],
    local_map: list[list[Any]],
) -> dict[str, Any]:
    cell_states: dict[tuple[int, int], Any] = {}
    victim_positions: list[tuple[int, int]] = []
    danger_zones: list[tuple[int, int]] = []
    explored_cells: list[tuple[int, int]] = []

    for row_index, row in enumerate(local_map):
        for col_index, cell in enumerate(row):
            if cell is None:
                continue

            absolute_position = (row_index, col_index)
            cell_states[absolute_position] = cell
            explored_cells.append(absolute_position)

            if cell == "victim":
                victim_positions.append(absolute_position)
            if cell in {"fire", "obstacle"}:
                danger_zones.append(absolute_position)

    return {
        "cell_states": cell_states,
        "victim_positions": victim_positions,
        "danger_zones": danger_zones,
        "agent_position": position,
        "energy": state["agent_energy"].get(agent_id, 100),
        "explored_cells": explored_cells,
        "assigned_victims": list(state.get("assigned_tasks", {}).values()),
    }


def _known_explored_cells(state: RescueState, agent_id: str) -> list[tuple[int, int]]:
    local_map = state["local_maps"].get(agent_id, [])
    explored: list[tuple[int, int]] = []
    for row_index, row in enumerate(local_map):
        for col_index, cell in enumerate(row):
            if cell is not None:
                explored.append((row_index, col_index))
    return explored


def _approach_cells(environment: Environment, victim_position: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = victim_position
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [candidate for candidate in candidates if environment._in_bounds(candidate) and environment.is_valid_move(candidate)]


def _is_adjacent_to_victim(environment: Environment, position: tuple[int, int], victim_position: tuple[int, int]) -> bool:
    row, col = position
    victim_row, victim_col = victim_position
    return abs(row - victim_row) + abs(col - victim_col) == 1


def _perform_rescue(state: RescueState, agent_id: str, victim_position: tuple[int, int]) -> None:
    environment = _require_environment()
    if victim_position in state["rescued_victims"]:
        return

    environment.set_cell(victim_position, "rescued")
    state["rescued_victims"].append(victim_position)
    state["assigned_tasks"].pop(agent_id, None)
    state["agent_energy"][agent_id] = max(0, state["agent_energy"].get(agent_id, 100) - 10)
    _send_message(
        state,
        sender=agent_id,
        receiver="Supervisor",
        message_type="rescue_complete",
        content=f"Rescued victim at {victim_position}",
    )
    _log_action(state, agent_id, f"mark_rescued_{victim_position}", state["mission_step"])


def _plan_a_star_path(
    environment: Environment,
    knowledge_base: KnowledgeBase,
    start: tuple[int, int],
    goals: list[tuple[int, int]],
    expand_danger_buffer: bool = False,
) -> list[tuple[int, int]]:
    goal_set = set(goals)
    open_heap: list[tuple[int, int, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0, 0, start))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], int] = {start: 0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in goal_set:
            return _reconstruct_path(came_from, current)

        for neighbor in _neighbors(environment, current):
            if _is_blocked_for_path(knowledge_base, neighbor, environment, expand_danger_buffer):
                continue

            tentative_cost = current_cost + 1
            if tentative_cost < g_score.get(neighbor, 10**9):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + min(_manhattan_distance(neighbor, goal) for goal in goal_set)
                heapq.heappush(open_heap, (priority, tentative_cost, neighbor))

    return [start]


def _neighbors(environment: Environment, position: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = position
    candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
    return [candidate for candidate in candidates if environment._in_bounds(candidate)]


def _is_blocked_for_path(
    knowledge_base: KnowledgeBase,
    position: tuple[int, int],
    environment: Environment,
    expand_danger_buffer: bool,
) -> bool:
    cell = environment.get_cell(position)
    if cell in {"obstacle", "fire"}:
        return True

    # Keep normal planning permissive and only harden constraints during replanning.
    if expand_danger_buffer:
        if position in knowledge_base.known_danger_zones:
            return True
        for neighbor in _neighbors(environment, position):
            if neighbor in knowledge_base.known_danger_zones:
                return True

    return False


def _fallback_path_toward_target(
    environment: Environment,
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    frontier: list[tuple[int, int]] = [start]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while frontier:
        current = frontier.pop(0)
        for neighbor in _neighbors(environment, current):
            if neighbor in parent:
                continue
            if not environment.is_valid_move(neighbor):
                continue
            parent[neighbor] = current
            frontier.append(neighbor)

    if not parent:
        return [start]

    best_cell = min(
        parent.keys(),
        key=lambda cell: (_manhattan_distance(cell, target), _manhattan_distance(start, cell)),
    )
    return _reconstruct_path(parent, best_cell)


def _path_contains_known_danger(knowledge_base: KnowledgeBase, path: list[tuple[int, int]]) -> bool:
    return any(position in knowledge_base.known_danger_zones for position in path[1:])


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path


def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
