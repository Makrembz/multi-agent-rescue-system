from __future__ import annotations

import random
from typing import Literal

CellType = Literal["empty", "obstacle", "fire", "victim", "safe", "rescued"]
ObservationCell = CellType | None
Position = tuple[int, int]


class Environment:
    """Grid-based disaster environment used by the rescue agents."""

    def __init__(
        self,
        n: int = 10,
        num_victims: int = 3,
        num_obstacles: int = 15,
        num_fires: int = 5,
    ) -> None:
        if n <= 0:
            raise ValueError("Grid size must be greater than zero.")

        self.n = n
        self.grid: list[list[CellType]] = [["empty" for _ in range(n)] for _ in range(n)]
        self.max_obstacle_ratio = 0.22
        self.max_fire_ratio = 0.15

        # Reserve a safe zone in the bottom-right corner as the default evacuation point.
        self.safe_position: Position = (n - 1, n - 1)
        self.grid[self.safe_position[0]][self.safe_position[1]] = "safe"

        # Keep spawn lanes reliable for a smoother simulation demo.
        self.protected_positions: set[Position] = {
            (0, 0),
            (1, 0),
            (0, n - 1),
            (1, n - 1),
            self.safe_position,
        }

        available_positions = [
            (row, col)
            for row in range(n)
            for col in range(n)
            if (row, col) != self.safe_position
        ]

        max_obstacles = int((n * n) * 0.15)
        max_fires = int((n * n) * 0.08)

        obstacle_count = min(num_obstacles, max_obstacles)
        fire_count = min(num_fires, max_fires)

        self._place_random_cells(available_positions, "obstacle", obstacle_count)
        self._place_random_cells(available_positions, "fire", fire_count)
        self._place_random_cells(available_positions, "victim", num_victims)
        self._ensure_victim_accessibility()

    def get_local_observation(
        self,
        position: Position,
        radius: int,
        noise: bool = True,
    ) -> list[list[ObservationCell]]:
        """Return the partially observed area centered on an agent position."""

        row, col = position
        observation: list[list[ObservationCell]] = []

        for current_row in range(row - radius, row + radius + 1):
            row_view: list[ObservationCell] = []
            for current_col in range(col - radius, col + radius + 1):
                if not self._in_bounds((current_row, current_col)):
                    row_view.append(None)
                    continue

                cell = self.grid[current_row][current_col]
                if noise:
                    cell = self._apply_noise(cell)
                row_view.append(cell)
            observation.append(row_view)

        return observation

    def update_environment(self) -> None:
        """Advance the simulation by spreading fire and creating collapses."""

        new_fire_positions: set[Position] = set()
        collapse_positions: set[Position] = set()
        total_cells = self.n * self.n
        current_fire = sum(1 for row in self.grid for cell in row if cell == "fire")
        current_obstacles = sum(1 for row in self.grid for cell in row if cell == "obstacle")

        fire_ratio = current_fire / total_cells
        obstacle_ratio = current_obstacles / total_cells

        spread_chance = 0.08 if fire_ratio < self.max_fire_ratio else 0.0

        for row in range(self.n):
            for col in range(self.n):
                if self.grid[row][col] != "fire":
                    continue

                for neighbor in self._neighbors((row, col)):
                    if neighbor in self.protected_positions:
                        continue
                    if self.grid[neighbor[0]][neighbor[1]] in {"empty", "victim"}:
                        if random.random() < spread_chance:
                            new_fire_positions.add(neighbor)

        for row in range(self.n):
            for col in range(self.n):
                position = (row, col)
                current_cell = self.grid[row][col]
                if position in self.protected_positions:
                    continue
                if current_cell in {"obstacle", "fire", "rescued", "victim", "safe"}:
                    continue

                if obstacle_ratio >= self.max_obstacle_ratio:
                    continue

                near_fire = any(self.grid[n_row][n_col] == "fire" for n_row, n_col in self._neighbors(position))
                collapse_chance = 0.02 if near_fire else 0.004
                if random.random() < collapse_chance:
                    collapse_positions.add(position)

        for row, col in new_fire_positions:
            self.grid[row][col] = "fire"

        for row, col in collapse_positions:
            if self.grid[row][col] != "fire":
                self.grid[row][col] = "obstacle"

        self._enforce_hazard_caps()
        self._ensure_victim_accessibility()

    def is_valid_move(self, position: Position) -> bool:
        """Check whether an agent can enter a given cell."""

        if not self._in_bounds(position):
            return False

        cell = self.get_cell(position)
        return cell in {"empty", "victim", "safe", "rescued"}

    def display(self) -> None:
        """Print the grid to the console using compact symbolic markers."""

        symbols = {
            "empty": ".",
            "obstacle": "O",
            "fire": "F",
            "victim": "V",
            "safe": "S",
            "rescued": "R",
        }

        print("Current environment:")
        print("  " + " ".join(str(index % 10) for index in range(self.n)))
        for row_index, row in enumerate(self.grid):
            row_symbols = " ".join(symbols[cell] for cell in row)
            print(f"{row_index % 10} {row_symbols}")

    def get_cell(self, position: Position) -> CellType:
        """Return the cell type at a given position."""

        self._validate_position(position)
        return self.grid[position[0]][position[1]]

    def set_cell(self, position: Position, cell_type: CellType) -> None:
        """Update the cell type at a given position."""

        self._validate_position(position)
        if cell_type not in {"empty", "obstacle", "fire", "victim", "safe", "rescued"}:
            raise ValueError(f"Unsupported cell type: {cell_type}")
        self.grid[position[0]][position[1]] = cell_type

    def _place_random_cells(
        self,
        available_positions: list[Position],
        cell_type: CellType,
        count: int,
    ) -> None:
        if count <= 0 or not available_positions:
            return

        number_to_place = min(count, len(available_positions))
        chosen_positions = random.sample(available_positions, number_to_place)

        for position in chosen_positions:
            self.grid[position[0]][position[1]] = cell_type
            available_positions.remove(position)

    def _apply_noise(self, cell: CellType) -> ObservationCell:
        if random.random() >= 0.10:
            return cell

        if cell == "empty":
            return random.choice(["obstacle", "fire", "victim"])

        return "empty"

    def _neighbors(self, position: Position) -> list[Position]:
        row, col = position
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        return [candidate for candidate in candidates if self._in_bounds(candidate)]

    def _in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.n and 0 <= col < self.n

    def _validate_position(self, position: Position) -> None:
        if not self._in_bounds(position):
            raise IndexError(f"Position out of bounds: {position}")

    def _ensure_victim_accessibility(self) -> None:
        victims: list[Position] = []
        for row in range(self.n):
            for col in range(self.n):
                if self.grid[row][col] == "victim":
                    victims.append((row, col))

        for victim in victims:
            neighbors = self._neighbors(victim)
            walkable_neighbors = [position for position in neighbors if self.grid[position[0]][position[1]] in {"empty", "safe", "rescued"}]
            if walkable_neighbors:
                continue

            for neighbor in neighbors:
                if neighbor == self.safe_position:
                    continue
                if self.grid[neighbor[0]][neighbor[1]] in {"obstacle", "fire"}:
                    self.grid[neighbor[0]][neighbor[1]] = "empty"
                    break

    def _enforce_hazard_caps(self) -> None:
        total_cells = self.n * self.n
        max_obstacles = int(total_cells * self.max_obstacle_ratio)
        max_fires = int(total_cells * self.max_fire_ratio)

        fire_cells = [(row, col) for row in range(self.n) for col in range(self.n) if self.grid[row][col] == "fire"]
        obstacle_cells = [(row, col) for row in range(self.n) for col in range(self.n) if self.grid[row][col] == "obstacle"]

        random.shuffle(fire_cells)
        random.shuffle(obstacle_cells)

        while len(fire_cells) > max_fires:
            row, col = fire_cells.pop()
            if (row, col) not in self.protected_positions:
                self.grid[row][col] = "empty"

        while len(obstacle_cells) > max_obstacles:
            row, col = obstacle_cells.pop()
            if (row, col) not in self.protected_positions:
                self.grid[row][col] = "empty"
