from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class MissionLogger:
    total_steps: int = 0
    victims_detected_per_step: dict[tuple[int, int], int] = field(default_factory=dict)
    victims_rescued_per_step: dict[tuple[int, int], int] = field(default_factory=dict)
    agent_energy_history: dict[str, list[int]] = field(default_factory=dict)
    messages_sent: int = 0
    tasks_assigned: int = 0
    path_replanning_count: int = 0
    action_log: list[dict[str, Any]] = field(default_factory=list)

    def log_step(self, state: dict[str, Any]) -> None:
        self.total_steps = int(state.get("mission_step", self.total_steps))

        for victim_position in state.get("detected_victims", []):
            self.victims_detected_per_step.setdefault(tuple(victim_position), self.total_steps)

        for victim_position in state.get("rescued_victims", []):
            self.victims_rescued_per_step.setdefault(tuple(victim_position), self.total_steps)

        for agent_id, energy in state.get("agent_energy", {}).items():
            self.agent_energy_history.setdefault(agent_id, []).append(int(energy))

        self.messages_sent = len(state.get("messages", []))
        self.tasks_assigned = sum(
            1 for record in state.get("action_log", []) if str(record.get("action", "")).startswith("assign_")
        )
        self.path_replanning_count = sum(
            1 for record in state.get("action_log", []) if "replan" in str(record.get("action", "")).lower()
        )
        self.action_log = list(state.get("action_log", []))

    def get_avg_detection_time(self) -> float:
        if not self.victims_detected_per_step:
            return 0.0
        return float(mean(self.victims_detected_per_step.values()))

    def get_avg_rescue_time(self) -> float:
        if not self.victims_rescued_per_step:
            return 0.0

        durations: list[int] = []
        for victim_position, rescue_step in self.victims_rescued_per_step.items():
            detected_step = self.victims_detected_per_step.get(victim_position)
            if detected_step is not None:
                durations.append(rescue_step - detected_step)

        if not durations:
            return 0.0
        return float(mean(durations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "victims_detected_per_step": {str(position): step for position, step in self.victims_detected_per_step.items()},
            "victims_rescued_per_step": {str(position): step for position, step in self.victims_rescued_per_step.items()},
            "agent_energy_history": self.agent_energy_history,
            "messages_sent": self.messages_sent,
            "tasks_assigned": self.tasks_assigned,
            "path_replanning_count": self.path_replanning_count,
            "avg_detection_time": self.get_avg_detection_time(),
            "avg_rescue_time": self.get_avg_rescue_time(),
            "action_log": self.action_log,
        }

    def export_json(self, filename: str = "mission_log.json") -> Path:
        path = Path(filename)
        path.write_text(json.dumps({"action_log": self.action_log}, indent=2), encoding="utf-8")
        return path
