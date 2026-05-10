import math

class BaseGame:
    """Parent class demonstrating inheritance and encapsulation."""
    def __init__(self, regions):
        self.regions = regions  # list of (x, y, w, h)
        self.found_regions = []
        self.mistakes = 0

    def is_game_over(self):
        """To be overridden by subclass (Polymorphism)"""
        raise NotImplementedError("Subclass must implement abstract method")

    def is_level_complete(self):
        """To be overridden by subclass (Polymorphism)"""
        raise NotImplementedError("Subclass must implement abstract method")

    def get_status(self):
        """To be overridden by subclass (Polymorphism)"""
        raise NotImplementedError("Subclass must implement abstract method")


class GameState(BaseGame):
    """Handles the core logic, tracking unfound differences and mistakes."""
    MAX_MISTAKES = 3
    PROXIMITY_RADIUS = 30

    def __init__(self, regions=None):
        super().__init__(regions if regions else [])
        self.cumulative_score = 0  # Tracked across multiple images [cite: 43]
        self.unfound_regions = list(self.regions)

    def reset(self, new_regions):
        """Resets the state for a newly loaded image while keeping the cumulative score."""
        self.regions = new_regions
        self.unfound_regions = list(new_regions)
        self.found_regions = []
        self.mistakes = 0

    def register_click(self, x, y):
        """Validates click against unfound regions."""
        if self.is_game_over() or self.is_level_complete():
            return "locked", None

        # Check if click is within reasonable proximity to an unfound difference [cite: 44]
        for region in self.unfound_regions:
            rx, ry, rw, rh = region
            
            # Calculate the center of the region for distance checking
            cx = rx + rw / 2
            cy = ry + rh / 2
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            
            if distance <= self.PROXIMITY_RADIUS:
                self.unfound_regions.remove(region)
                self.found_regions.append(region)
                self.cumulative_score += 1
                
                if self.is_level_complete():
                    return "level_complete", region
                return "found", region
                
        # If no region matched, count as mistake [cite: 49]
        self.mistakes += 1
        if self.is_game_over():
            return "game_over", None
        return "mistake", None

    def reveal_all(self):
        """Moves all unfound regions to revealed state."""
        revealed = list(self.unfound_regions)
        self.unfound_regions.clear()
        return revealed

    def is_game_over(self):
        return self.mistakes >= self.MAX_MISTAKES

    def is_level_complete(self):
        return len(self.unfound_regions) == 0 and len(self.regions) > 0

    def get_status(self):
        return {
            "remaining": len(self.unfound_regions),
            "mistakes": self.mistakes,
            "score": self.cumulative_score
        }