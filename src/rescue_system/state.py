from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

CellType = Literal["empty", "obstacle", "fire", "safe", "victim", "rescued"]
CellObservation = CellType | None
Position = tuple[int, int]


class MessageRecord(TypedDict):
    # Agent that sent the message.
    sender: str
    # Agent that should receive or inspect the message.
    receiver: str
    # Message category, such as status, alert, task, or request.
    type: str
    # Payload of the message with the operational detail.
    content: str


class ActionRecord(TypedDict):
    # Agent that performed the action.
    agent: str
    # Action description, such as move, scan, rescue, or report.
    action: str
    # Simulation step or timestamp when the action happened.
    step: int


class RescueState(TypedDict):
    # The NxN disaster-zone map shared by all agents.
    grid: list[list[CellType]]
    # Current row/column position of each agent in the environment.
    agent_positions: dict[str, Position]
    # Remaining energy level for each agent, stored as a percentage.
    agent_energy: dict[str, int]
    # Partial map observed by each agent from its own perspective.
    local_maps: dict[str, list[list[CellObservation]]]
    # Victim locations that have been confirmed by one or more agents.
    detected_victims: list[Position]
    # Victim assignment for each robot, mapped by robot id.
    assigned_tasks: dict[str, Position]
    # Victim locations that have already been rescued.
    rescued_victims: list[Position]
    # Messages exchanged between agents during the mission.
    messages: list[MessageRecord]
    # Step-by-step operational history of all actions taken in the mission.
    action_log: list[ActionRecord]
    # Current simulation step counter.
    mission_step: int
    # True when the rescue mission has reached a terminal state.
    mission_complete: bool
