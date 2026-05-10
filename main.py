import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from game_state import GameState
import math


# ─────────────────────────────────────────────
#  Main application window
#  Inherits from tk.Tk so it IS the root window
# ─────────────────────────────────────────────
class GameWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Find the Differences")
        self.geometry("1000x600")

        # Game logic lives here; UI just reads from / writes to it
        self.game_state = GameState()
        self._build_ui()

    # ------------------------------------------------------------------
    #  Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Top bar: buttons on the left, status text on the right ──
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.load_btn = tk.Button(
            self.top_frame, text="Load Image", command=self.load_image
        )
        self.load_btn.pack(side=tk.LEFT, padx=10)

        self.reveal_btn = tk.Button(
            self.top_frame, text="Reveal", command=self.reveal_differences
        )
        self.reveal_btn.pack(side=tk.LEFT, padx=10)

        self.status_lbl = tk.Label(
            self.top_frame,
            text="Remaining: 0 | Mistakes: 0/3 | Score: 0",
            font=("Arial", 12),
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # ── Image area: two canvases sitting side by side ──
        self.img_frame = tk.Frame(self)
        self.img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas_left = tk.Canvas(self.img_frame, bg="gray")
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas_right = tk.Canvas(self.img_frame, bg="gray")
        self.canvas_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Only the RIGHT canvas is clickable — that's the "modified" image
        self.canvas_right.bind("<Button-1>", self.on_modified_click)

        # These will hold PhotoImage references so they aren't garbage-collected
        self.photo_left = None
        self.photo_right = None

    # ------------------------------------------------------------------
    #  Loading & displaying images
    # ------------------------------------------------------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp")]
        )
        if not file_path:
            return  # user hit cancel

        # ── Swap this block in once ImageProcessor (Task 1) is ready ──
        # processor = ImageProcessor(file_path)
        # processor.load_image()
        # processor.clone_image()
        # regions = processor.generate_differences()
        # self.original_img_cv = processor.original
        # self.modified_img_cv = processor.modified

        # ── MOCK DATA — lets us test the UI without real images ──
        # Both images are plain gray; differences are just pretend rectangles
        self.original_img_cv = np.zeros((400, 400, 3), dtype=np.uint8)
        self.original_img_cv[:] = (200, 200, 200)       # fill with gray
        self.modified_img_cv = self.original_img_cv.copy()

        # Each tuple is (x, y, width, height) of a difference region
        regions = [
            (50,  50,  40, 40),
            (150, 100, 30, 30),
            (300, 200, 40, 40),
            (100, 300, 30, 30),
            (250, 300, 40, 40),
        ]
        # ─────────────────────────────────────────────────────────────

        self.game_state.reset(regions)
        self.update_status_bar()
        self.display_images()

    def display_images(self):
        # OpenCV stores pixels as BGR; PIL / tkinter expects RGB — flip it
        orig_rgb = cv2.cvtColor(self.original_img_cv, cv2.COLOR_BGR2RGB)
        mod_rgb  = cv2.cvtColor(self.modified_img_cv,  cv2.COLOR_BGR2RGB)

        orig_pil = Image.fromarray(orig_rgb)
        mod_pil  = Image.fromarray(mod_rgb)

        # Keep references on self — if they go out of scope the canvas goes blank
        self.photo_left  = ImageTk.PhotoImage(orig_pil)
        self.photo_right = ImageTk.PhotoImage(mod_pil)

        # Clear old drawings before placing the new image
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")

        self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.photo_left)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.photo_right)

    # ------------------------------------------------------------------
    #  Click handling
    # ------------------------------------------------------------------
    def on_modified_click(self, event):
        # Nothing to click on if no image has been loaded yet
        if not self.game_state.regions:
            return

        status, region = self.game_state.register_click(event.x, event.y)
        self.update_status_bar()

        if status in ("found", "level_complete"):
            # Draw a red circle centred on the difference region
            rx, ry, rw, rh = region
            cx = rx + rw / 2
            cy = ry + rh / 2
            self.draw_circle(cx, cy, "red")

            if status == "level_complete":
                messagebox.showinfo(
                    "Success", "You found all 5 differences! Load a new image."
                )

        elif status == "game_over":
            messagebox.showerror(
                "Game Over", "3 Mistakes Reached! No further guesses allowed."
            )

    # ------------------------------------------------------------------
    #  Drawing helpers
    # ------------------------------------------------------------------
    def draw_circle(self, x, y, color):
        r = GameState.PROXIMITY_RADIUS

        # Mirror the circle on both canvases so the player can compare easily
        for canvas in (self.canvas_left, self.canvas_right):
            canvas.create_oval(
                x - r, y - r, x + r, y + r,
                outline=color, width=3
            )

    # ------------------------------------------------------------------
    #  UI updates
    # ------------------------------------------------------------------
    def update_status_bar(self):
        stats = self.game_state.get_status()
        self.status_lbl.config(
            text=(
                f"Remaining: {stats['remaining']} | "
                f"Mistakes: {stats['mistakes']}/{GameState.MAX_MISTAKES} | "
                f"Score: {stats['score']}"
            )
        )

    def reveal_differences(self):
        # Nothing to reveal if no image is loaded or the level is already done
        if not self.game_state.regions or self.game_state.is_level_complete():
            return

        # Mark every undiscovered region in the game state and get them back
        revealed_regions = self.game_state.reveal_all()

        for region in revealed_regions:
            rx, ry, rw, rh = region
            cx = rx + rw / 2
            cy = ry + rh / 2
            # Blue = revealed by the player giving up; red = found by clicking
            self.draw_circle(cx, cy, "blue")

        self.update_status_bar()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = GameWindow()
    app.mainloop()