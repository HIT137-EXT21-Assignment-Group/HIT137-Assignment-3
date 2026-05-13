"""
HIT137 Assignment 3 — Game State
Demonstrates OOP: inheritance, encapsulation, polymorphism.
"""

import math


# ─────────────────────────────────────────────────────────────────────
#  Base class — encapsulates shared game state and abstract interface
# ─────────────────────────────────────────────────────────────────────
class BaseGame:
    """
    Parent class demonstrating inheritance and encapsulation.
    Defines the interface (polymorphic methods) that every game type
    must implement.
    """

    def __init__(self, regions: list):
        # Encapsulated state — child classes extend, not replace
        self.regions       = regions      # all difference regions for this round
        self.found_regions = []           # correctly identified regions
        self.mistakes      = 0            # wrong-click counter

    # ── Polymorphic interface (overridden by GameState) ──────────────

    def is_game_over(self) -> bool:
        """Returns True when the player has used all allowed mistakes."""
        raise NotImplementedError("Subclass must implement is_game_over()")

    def is_level_complete(self) -> bool:
        """Returns True when every difference has been found."""
        raise NotImplementedError("Subclass must implement is_level_complete()")

    def get_status(self) -> dict:
        """Returns a snapshot of current game metrics."""
        raise NotImplementedError("Subclass must implement get_status()")


# ─────────────────────────────────────────────────────────────────────
#  GameState — concrete game logic
# ─────────────────────────────────────────────────────────────────────
class GameState(BaseGame):
    """
    Handles all game logic:
      - Proximity-based click validation
      - Mistake tracking and lockout
      - Per-round and cumulative scoring
      - Reveal-all shortcut
    """

    MAX_MISTAKES     = 3
    PROXIMITY_RADIUS = 30   # pixels — how close a click must be to count

    def __init__(self, regions: list = None):
        super().__init__(regions if regions else [])
        # Cumulative score persists across multiple loaded images
        self.cumulative_score = 0
        self.unfound_regions  = list(self.regions)

    # ── Round management ─────────────────────────────────────────────

    def reset(self, new_regions: list):
        """
        Resets per-round state for a newly loaded image.
        Cumulative score is intentionally preserved.
        """
        self.regions         = new_regions
        self.unfound_regions = list(new_regions)
        self.found_regions   = []
        self.mistakes        = 0
        # cumulative_score is NOT reset — carries over between images

    # ── Click handling ───────────────────────────────────────────────

    def register_click(self, x: int, y: int) -> tuple:
        """
        Validates a player click against all unfound regions.

        Returns
        -------
        ("locked",         None)   — game over or level already complete
        ("found",          region) — correct click, more to find
        ("level_complete", region) — correct click, all differences found
        ("game_over",      None)   — wrong click that exhausts mistakes
        ("mistake",        None)   — wrong click, mistakes remaining
        """
        if self.is_game_over() or self.is_level_complete():
            return "locked", None

        for region in self.unfound_regions:
            rx, ry, rw, rh = region
            cx = rx + rw / 2
            cy = ry + rh / 2
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            if distance <= self.PROXIMITY_RADIUS:
                self.unfound_regions.remove(region)
                self.found_regions.append(region)
                self.cumulative_score += 1

                if self.is_level_complete():
                    return "level_complete", region
                return "found", region

        # No region matched — count as a mistake
        self.mistakes += 1
        if self.is_game_over():
            return "game_over", None
        return "mistake", None

    # ── Reveal shortcut ──────────────────────────────────────────────

    def reveal_all(self) -> list:
        """
        Moves every unfound region into the revealed bucket and returns them
        so the GUI can draw blue circles.  Remaining counter becomes 0.
        """
        revealed = list(self.unfound_regions)
        self.unfound_regions.clear()
        return revealed

    # ── Polymorphic overrides ─────────────────────────────────────────

    def is_game_over(self) -> bool:
        return self.mistakes >= self.MAX_MISTAKES

    def is_level_complete(self) -> bool:
        return len(self.unfound_regions) == 0 and len(self.regions) > 0

    def get_status(self) -> dict:
        """
        Returns all metrics the status bar and messages need.
        'found' and 'total' are included so the game-over prompt can
        report how many differences the player managed to find.
        """
        return {
            "remaining": len(self.unfound_regions),
            "found"    : len(self.found_regions),
            "total"    : len(self.regions),
            "mistakes" : self.mistakes,
            "score"    : self.cumulative_score,
        }