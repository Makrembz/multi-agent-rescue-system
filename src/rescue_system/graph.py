from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import drone1_node, drone2_node, register_environment, robot1_node, robot2_node
from .environment import Environment
from .state import CellType, RescueState
from .supervisor import supervisor_node, supervisor_router


def initialize_state() -> RescueState:
    """Create the initial rescue state and register a fresh environment."""

    environment = Environment(n=20, num_victims=5, num_obstacles=10, num_fires=5)
    register_environment(environment)

    _prepare_spawn_cells(environment)

    return {
        "grid": environment.grid,
        "agent_positions": {
            "Drone1": (0, 0),
            "Drone2": (1, 0),
            "Robot1": (0, environment.n - 1),
            "Robot2": (1, environment.n - 1),
        },
        "agent_energy": {
            "Drone1": 100,
            "Drone2": 100,
            "Robot1": 100,
            "Robot2": 100,
        },
        "local_maps": {
            "Drone1": [[None for _ in range(environment.n)] for _ in range(environment.n)],
            "Drone2": [[None for _ in range(environment.n)] for _ in range(environment.n)],
            "Robot1": [[None for _ in range(environment.n)] for _ in range(environment.n)],
            "Robot2": [[None for _ in range(environment.n)] for _ in range(environment.n)],
        },
        "detected_victims": [],
        "assigned_tasks": {},
        "rescued_victims": [],
        "messages": [],
        "action_log": [],
        "mission_step": 0,
        "mission_complete": False,
    }


def build_graph() -> StateGraph[RescueState]:
    """Assemble the rescue workflow graph."""

    graph = StateGraph(RescueState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("drone1", drone1_node)
    graph.add_node("drone2", drone2_node)
    graph.add_node("robot1", robot1_node)
    graph.add_node("robot2", robot2_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _supervisor_parallel_dispatch,
        {
            "continue": "drone1",
            "end": END,
        },
    )

    graph.add_edge("drone1", "drone2")
    graph.add_edge("drone2", "robot1")
    graph.add_edge("robot1", "robot2")
    graph.add_edge("robot2", "supervisor")

    return graph


def _supervisor_dispatch(state: RescueState) -> str:
    return supervisor_router(state)


def _supervisor_parallel_dispatch(state: RescueState) -> str:
    route = supervisor_router(state)
    return "end" if route == "end" else "continue"


def _prepare_spawn_cells(environment: Environment) -> None:
    spawn_positions = [(0, 0), (1, 0), (0, environment.n - 1), (1, environment.n - 1)]
    for position in spawn_positions:
        if environment._in_bounds(position):
            environment.set_cell(position, "empty")
    environment.set_cell(environment.safe_position, "safe")


graph = build_graph()
app = graph.compile()


def show_graph_structure() -> str:
    """Return a compact textual description of the compiled graph structure."""

    return "\n".join(
        [
            "START -> supervisor",
            "supervisor -> drone1 | continue | END",
            "drone1 -> drone2",
            "drone2 -> robot1",
            "robot1 -> robot2",
            "robot2 -> supervisor",
        ]
    )
