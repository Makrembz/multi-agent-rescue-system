# Multi-Agent Rescue System

A Python simulation of coordinated disaster response using a supervisor, two drones, and two ground robots.

The system models a dynamic grid environment with hazards (fire and obstacles), victim detection, task allocation, path planning, and mission analytics. A Streamlit dashboard provides live control and visualization.

## Features

- Multi-agent coordination workflow powered by LangGraph
- Dynamic 2D environment with:
  - spreading fire
  - random collapses (new obstacles)
  - victim accessibility safeguards
- Two drone agents for exploration and victim detection
- Two robot agents for rescue execution with A* path planning
- Supervisor agent for task assignment and mission completion logic
- Rule-based knowledge base for reasoning (energy, danger, victim priority)
- Live Streamlit dashboard with:
  - map rendering
  - agent status and energy bars
  - mission progress metrics
  - action log and assignment table
  - mission report with charts and JSON export

## Project Structure

```text
multi-agent-rescue-system/
├── dashboard.py
├── mission_log.json
├── requirements.txt
└── src/
    └── rescue_system/
        ├── __init__.py
        ├── environment.py
        ├── graph.py
        ├── knowledge_base.py
        ├── mission_logger.py
        ├── state.py
        ├── supervisor.py
        └── agents/
            ├── __init__.py
            ├── drone.py
            └── robot.py
```

## Architecture Overview

The execution loop follows this pattern:

1. Supervisor reads state, ingests drone messages, and assigns victims to available robots.
2. Drones explore, update local maps, and report confirmed victims.
3. Robots move toward assigned victims and perform rescues when adjacent.
4. Supervisor evaluates mission completion.
5. Environment evolves (fire spread / collapses), and the mission logger records metrics.

LangGraph graph flow:

```text
START -> supervisor
supervisor -> drone1 -> drone2 -> robot1 -> robot2 -> supervisor
supervisor -> END (when mission complete)
```

## System Architecture

```text
┌─────────────────────────────────────────────────┐
│              LANGGRAPH SUPERVISOR               │
│         Orchestration · Routing · Tasks         │
└──────────┬──────────────────────┬───────────────┘
      │                      │
    ┌──────▼──────┐        ┌──────▼──────┐
    │   DRONES    │        │   ROBOTS    │
    │   Drone 1   │        │   Robot 1   │
    │   Drone 2   │        │   Robot 2   │
    └──────┬──────┘        └──────┬──────┘
      │                      │
    ┌──────▼──────────────────────▼──────┐
    │             SHARED STATE           │
    │   Grid · Positions · Messages · KB │
    └────────────────┬───────────────────┘
      │
    ┌────────────────▼───────────────────┐
    │         ENVIRONMENT (POMDP)        │
    │    NxN Grid · Fire · Obstacles ·   │
    │    Victims · Partial Observability │
    └────────────────────────────────────┘
```

## State Model

Core state fields include:

- `grid`: global NxN cell map
- `agent_positions`: position of each agent
- `agent_energy`: per-agent battery level
- `local_maps`: per-agent partial observations
- `detected_victims`: confirmed victim coordinates
- `assigned_tasks`: robot -> victim assignments
- `rescued_victims`: rescued victim coordinates
- `messages`: inter-agent communication records
- `action_log`: chronological mission actions
- `mission_step`: simulation step counter
- `mission_complete`: terminal mission flag

## Agent Behaviors

### Supervisor

- Parses incoming victim alerts from messages
- Assigns nearest available robot (energy > 10)
- Cleans stale assignments after rescues
- Marks mission complete when no victims remain and no assignments are pending

### Drones (Drone1, Drone2)

- Observe local cells with configurable radius and noisy sensing
- Follow complementary snake-like exploration patterns
- Avoid overlapping too closely with teammate
- Confirm victims against environment truth before reporting
- Consume energy on movement and pause when low

### Robots (Robot1, Robot2)

- Stay idle until assigned by supervisor
- Build belief state from local map + assignment data
- Plan routes using A* to safe approach cells adjacent to a victim
- Replan if known danger contaminates path
- Use fallback BFS-like progression if ideal path is unavailable
- Rescue victim when adjacent and report completion

## Environment Rules

- Cell types: `empty`, `obstacle`, `fire`, `victim`, `safe`, `rescued`
- Protected spawn/safe cells reduce early deadlocks
- Fire may spread to nearby `empty` or `victim` cells
- Random collapses can turn cells into `obstacle`
- Hazard caps prevent runaway fire/obstacle dominance
- Victim accessibility pass prevents fully sealed victims

## Dashboard

Run and control simulation from Streamlit UI:

- `Next Step`: advance one cycle
- `Run Auto`: continuous stepping
- `Pause Auto`: stop continuous stepping
- `Reset`: regenerate environment and state

Views include:

- Live grid with drone/robot markers
- Agent status cards and energy bars
- Detection and rescue counters
- Task assignment table
- Recent action log
- Mission report tab with:
  - energy trends
  - detection-to-rescue timeline
  - summary metrics
  - export of `mission_log.json`

## Requirements

From `requirements.txt`:

- streamlit >= 1.33
- matplotlib >= 3.8
- langgraph >= 0.2

Python 3.10+ is recommended.

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run

```powershell
streamlit run dashboard.py
```

Then open the local URL shown by Streamlit (usually http://localhost:8501).

## How Mission Logging Works

The `MissionLogger` tracks:

- total steps
- victim detection and rescue step indices
- per-agent energy history
- message count
- number of assignments and replans
- complete action log snapshot

Export behavior:

- UI export button downloads a JSON report
- `export_json()` writes an `action_log` payload to `mission_log.json`

## Customization Ideas

- Tune environment size and hazard/victim counts in `initialize_state()`
- Adjust drone vision radius and movement costs
- Refine supervisor routing and assignment policy
- Add richer communication protocols (priority queues, confidence scoring)
- Add benchmark scenarios and reproducible random seeds

## Troubleshooting

### Streamlit command not found

Use:

```powershell
python -m streamlit run dashboard.py
```

### Empty or stalled behavior

Use the dashboard Debug Checks expander to inspect:

- whether supervisor routes to robots
- whether assignments are being created
- whether robot nodes are executing


