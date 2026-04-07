from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CellType = Literal["empty", "obstacle", "fire", "safe", "victim", "rescued"]
Position = tuple[int, int]


@dataclass
class KnowledgeBase:
    """Stores agent facts, rules, and simple forward-chaining conclusions."""

    # Known cell states keyed by position.
    known_cell_states: dict[Position, CellType] = field(default_factory=dict)
    # Confirmed victim locations.
    known_victim_positions: set[Position] = field(default_factory=set)
    # Confirmed danger zones such as fire and obstacle cells.
    known_danger_zones: set[Position] = field(default_factory=set)
    # Current position of the agent using this knowledge base.
    agent_position: Position | None = None
    # Optional remaining energy value for the agent.
    agent_energy: int | None = None
    # Cells or zones that the agent has already explored.
    explored_cells: set[Position] = field(default_factory=set)
    # Victims that should be prioritized for rescue.
    priority_victims: set[Position] = field(default_factory=set)
    # Victims that already have a robot assigned.
    assigned_victims: set[Position] = field(default_factory=set)
    # Destination returned by the stop-and-retreat rule.
    return_to_base: bool = False
    # Suggested unexplored direction or zone when local area is exhausted.
    move_to_unexplored_zone: bool = False
    # Latest conclusions produced by the inference engine.
    conclusions: list[dict[str, Any]] = field(default_factory=list)

    def tell(self, fact_type: str, value: Any) -> None:
        """Add or update a fact in the knowledge base."""

        if fact_type == "cell_state":
            position, cell_type = value
            self.known_cell_states[position] = cell_type
            if cell_type in {"fire", "obstacle"}:
                self.known_danger_zones.add(position)
            if cell_type == "victim":
                self.known_victim_positions.add(position)
        elif fact_type == "victim_position":
            self.known_victim_positions.add(value)
        elif fact_type == "danger_zone":
            self.known_danger_zones.add(value)
        elif fact_type == "agent_position":
            self.agent_position = value
        elif fact_type == "energy":
            self.agent_energy = int(value)
        elif fact_type == "explored_cell":
            self.explored_cells.add(value)
        elif fact_type == "priority_victim":
            self.priority_victims.add(value)
        elif fact_type == "assigned_victim":
            self.assigned_victims.add(value)
        else:
            raise ValueError(f"Unsupported fact type: {fact_type}")

    def ask(self, fact_type: str) -> Any:
        """Query a fact or derived knowledge from the knowledge base."""

        if fact_type == "cell_states":
            return dict(self.known_cell_states)
        if fact_type == "victim_positions":
            return set(self.known_victim_positions)
        if fact_type == "danger_zones":
            return set(self.known_danger_zones)
        if fact_type == "agent_position":
            return self.agent_position
        if fact_type == "energy":
            return self.agent_energy
        if fact_type == "priority_victims":
            return set(self.priority_victims)
        if fact_type == "assigned_victims":
            return set(self.assigned_victims)
        if fact_type == "return_to_base":
            return self.return_to_base
        if fact_type == "move_to_unexplored_zone":
            return self.move_to_unexplored_zone

        raise ValueError(f"Unsupported fact type: {fact_type}")

    def infer(self) -> list[dict[str, Any]]:
        """Run forward chaining and return the newly derived conclusions."""

        self.conclusions = []
        changed = True

        while changed:
            changed = False

            if self._rule_fire_implies_danger():
                changed = True
            if self._rule_victim_priority():
                changed = True
            if self._rule_low_energy_return():
                changed = True
            if self._rule_move_to_unexplored_zone():
                changed = True

        return list(self.conclusions)

    def update_belief_state(self, observation: dict[str, Any]) -> None:
        """Update facts using a local observation dictionary from an agent node."""

        cell_states = observation.get("cell_states", {})
        for position, cell_type in cell_states.items():
            self.tell("cell_state", (position, cell_type))

        for position in observation.get("victim_positions", []):
            self.tell("victim_position", position)

        for position in observation.get("assigned_victims", []):
            self.tell("assigned_victim", position)

        for position in observation.get("danger_zones", []):
            self.tell("danger_zone", position)

        if "agent_position" in observation:
            self.tell("agent_position", observation["agent_position"])

        if "energy" in observation:
            self.tell("energy", observation["energy"])

        for position in observation.get("explored_cells", []):
            self.tell("explored_cell", position)

    def _append_conclusion(self, rule: str, value: Any) -> bool:
        conclusion = {"rule": rule, "value": value}
        if conclusion in self.conclusions:
            return False
        self.conclusions.append(conclusion)
        return True

    def _rule_fire_implies_danger(self) -> bool:
        changed = False
        for position, cell_type in self.known_cell_states.items():
            if cell_type == "fire" and position not in self.known_danger_zones:
                self.known_danger_zones.add(position)
                changed = self._append_conclusion("fire_implies_danger", position) or changed
        return changed

    def _rule_victim_priority(self) -> bool:
        changed = False
        for position in self.known_victim_positions:
            if position in self.assigned_victims:
                continue
            if position not in self.priority_victims:
                self.priority_victims.add(position)
                changed = self._append_conclusion("victim_detected_priority", position) or changed
        return changed

    def _rule_low_energy_return(self) -> bool:
        if self.agent_energy is None or self.agent_energy >= 20:
            return False

        if self.return_to_base:
            return False

        self.return_to_base = True
        return self._append_conclusion("low_energy_return_to_base", self.agent_energy)

    def _rule_move_to_unexplored_zone(self) -> bool:
        if self.agent_position is None:
            return False

        adjacent_positions = self._adjacent_positions(self.agent_position)
        if not adjacent_positions:
            return False

        if all(position in self.explored_cells for position in adjacent_positions):
            if self.move_to_unexplored_zone:
                return False
            self.move_to_unexplored_zone = True
            return self._append_conclusion("adjacent_explored_move_unexplored", self.agent_position)

        return False

    def _adjacent_positions(self, position: Position) -> list[Position]:
        row, col = position
        return [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]
