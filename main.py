"""
HIT137 Assignment 3 — Main GUI
Tkinter application:  two-canvas spot-the-difference game.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from game_state import GameState
from image_processing import ImageProcessor


# ─────────────────────────────────────────────────────────────────────
#  GameWindow — the application window (inherits tk.Tk)
# ─────────────────────────────────────────────────────────────────────
class GameWindow(tk.Tk):
    """
    Main application class.

    Responsibilities
    ----------------
    - Build and manage all Tkinter widgets
    - Delegate image loading / processing to ImageProcessor
    - Delegate click validation / scoring to GameState
    - Draw feedback (red / blue circles) on both canvases
    - Keep the status bar synchronised with GameState

    OOP notes
    ---------
    Inherits tk.Tk so this IS the root window (no separate root needed).
    Interacts with GameState and ImageProcessor via clear method calls
    (class interaction / encapsulation).
    """

    # Minimum canvas size supplied to ImageProcessor when the window is
    # first opened (before the user resizes it).
    _MIN_CANVAS_W = 450
    _MIN_CANVAS_H = 480

    def __init__(self):
        super().__init__()
        self.title("Find the Differences")
        self.geometry("1100x640")
        self.minsize(800, 500)

        # Game logic — UI reads from / writes to this object
        self.game_state = GameState()

        # Image offset: distance from canvas origin to image top-left
        # (used to centre the image and adjust click coordinates)
        self._img_offset_x = 0
        self._img_offset_y = 0

        # Strong references keep PhotoImages alive (prevents blank canvas)
        self.photo_left  = None
        self.photo_right = None

        self._build_ui()

    # ── LAYOUT ────────────────────────────────────────────────────────

    def _build_ui(self):
        """Construct all widgets."""

        # ── Top bar ─────────────────────────────────────────────────
        top = tk.Frame(self, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="Load Image", width=12,
                  command=self.load_image).pack(side=tk.LEFT, padx=10)

        tk.Button(top, text="Reveal", width=10,
                  command=self.reveal_differences).pack(side=tk.LEFT, padx=4)

        self.status_lbl = tk.Label(
            top,
            text="Load an image to start  |  Remaining: –  |  "
                 "Mistakes: 0/3  |  Score: 0",
            font=("Arial", 11),
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # ── Canvas area ──────────────────────────────────────────────
        img_frame = tk.Frame(self)
        img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Labels above each canvas so the player knows which is which
        left_col = tk.Frame(img_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(left_col, text="Original", font=("Arial", 10, "bold")
                 ).pack(side=tk.TOP)
        self.canvas_left = tk.Canvas(left_col, bg="#4a4a4a",
                                     highlightthickness=1,
                                     highlightbackground="#888")
        self.canvas_left.pack(fill=tk.BOTH, expand=True)

        right_col = tk.Frame(img_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(right_col, text="Modified  (click to find differences)",
                 font=("Arial", 10, "bold")).pack(side=tk.TOP)
        self.canvas_right = tk.Canvas(right_col, bg="#4a4a4a",
                                      highlightthickness=1,
                                      highlightbackground="#888",
                                      cursor="crosshair")
        self.canvas_right.pack(fill=tk.BOTH, expand=True)

        # Only the modified (right) canvas accepts player clicks
        self.canvas_right.bind("<Button-1>", self.on_modified_click)

        # ── Bottom status bar ─────────────────────────────────────────
        self.bottom_lbl = tk.Label(
            self, text="", font=("Arial", 10), fg="#555"
        )
        self.bottom_lbl.pack(side=tk.BOTTOM, pady=4)

    # ── IMAGE LOADING ────────────────────────────────────────────────

    def load_image(self):
        """
        Open a file dialog, hand the chosen path to ImageProcessor,
        reset GameState for the new round, and display both images.
        Supports JPG, JPEG, PNG, BMP.
        """
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                       ("All files", "*.*")]
        )
        if not file_path:
            return  # user cancelled

        # Read actual canvas dimensions so the image fills the available space
        self.update_idletasks()
        canvas_w = max(self.canvas_left.winfo_width(),  self._MIN_CANVAS_W)
        canvas_h = max(self.canvas_left.winfo_height(), self._MIN_CANVAS_H)

        try:
            processor = ImageProcessor(
                display_width=canvas_w,
                display_height=canvas_h,
            )
            orig_display, mod_display, regions = \
                processor.load_image_from_path(file_path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        # Store scaled arrays for display
        self._orig_cv = orig_display
        self._mod_cv  = mod_display

        # GameState expects plain (x, y, w, h) 4-tuples
        # ImageProcessor returns 5-tuples that include alteration_type
        regions_4 = [(x, y, w, h) for (x, y, w, h, _) in regions]
        self.game_state.reset(regions_4)

        self.update_status_bar()
        self.bottom_lbl.config(text="")
        self.display_images()

    def display_images(self):
        """
        Convert OpenCV BGR arrays to Tkinter PhotoImages and draw them
        centred on both canvases.  Saves the image offset so click
        coordinates can be translated back to image space.
        """
        self.update_idletasks()

        orig_rgb = cv2.cvtColor(self._orig_cv, cv2.COLOR_BGR2RGB)
        mod_rgb  = cv2.cvtColor(self._mod_cv,  cv2.COLOR_BGR2RGB)

        # Keep strong references — without these the canvas goes blank
        self.photo_left  = ImageTk.PhotoImage(Image.fromarray(orig_rgb))
        self.photo_right = ImageTk.PhotoImage(Image.fromarray(mod_rgb))

        img_w = self.photo_left.width()
        img_h = self.photo_left.height()

        # Centre the image within the canvas
        cw = self.canvas_left.winfo_width()
        ch = self.canvas_left.winfo_height()
        self._img_offset_x = max(0, (cw - img_w) // 2)
        self._img_offset_y = max(0, (ch - img_h) // 2)

        ox, oy = self._img_offset_x, self._img_offset_y

        self.canvas_left.delete("all")
        self.canvas_right.delete("all")

        self.canvas_left.create_image( ox, oy, anchor=tk.NW, image=self.photo_left)
        self.canvas_right.create_image(ox, oy, anchor=tk.NW, image=self.photo_right)

    # ── CLICK HANDLING ───────────────────────────────────────────────

    def on_modified_click(self, event):
        """
        Handle a player click on the modified (right) canvas.

        1. Translate canvas coordinates → image coordinates
           (subtract the centering offset applied in display_images).
        2. Pass to GameState for validation.
        3. Draw red circle on both canvases if correct.
        4. Show appropriate prompt for level_complete / game_over.
        """
        if not self.game_state.regions:
            return  # no image loaded yet

        # Adjust for image centering offset
        img_x = event.x - self._img_offset_x
        img_y = event.y - self._img_offset_y

        status, region = self.game_state.register_click(img_x, img_y)
        self.update_status_bar()

        if status == "locked":
            return  # game already over or level complete — ignore

        if status in ("found", "level_complete"):
            rx, ry, rw, rh = region
            self.draw_circle(rx + rw / 2, ry + rh / 2, "red")

            if status == "level_complete":
                stats = self.game_state.get_status()
                messagebox.showinfo(
                    "Level Complete! 🎉",
                    f"You found all 5 differences!\n"
                    f"Cumulative score: {stats['score']}\n\n"
                    "Load another image to keep playing."
                )

        elif status == "mistake":
            stats = self.game_state.get_status()
            remaining_mistakes = GameState.MAX_MISTAKES - stats["mistakes"]
            self.bottom_lbl.config(
                text=f"Wrong! {remaining_mistakes} mistake(s) remaining.",
                fg="orange"
            )

        elif status == "game_over":
            stats = self.game_state.get_status()
            self.bottom_lbl.config(
                text="Game Over — no more guesses. Use Reveal or load a new image.",
                fg="red"
            )
            messagebox.showerror(
                "Game Over",
                f"3 mistakes reached!\n"
                f"You found {stats['found']} of {stats['total']} "
                f"differences.\n\n"
                "Press Reveal to see the remaining differences,\n"
                "or load a new image to restart."
            )

    # ── DRAWING ──────────────────────────────────────────────────────

    def draw_circle(self, img_x: float, img_y: float, color: str):
        """
        Draw a circle on BOTH canvases centred on the given image-space
        coordinates.  Adds the canvas offset so circles align with the
        centred image.

        color : "red"  for player-found differences
                "blue" for differences revealed by the Reveal button
        """
        r  = GameState.PROXIMITY_RADIUS
        ox = self._img_offset_x
        oy = self._img_offset_y
        cx = img_x + ox
        cy = img_y + oy

        for canvas in (self.canvas_left, self.canvas_right):
            canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=color, width=3
            )

    # ── STATUS BAR ───────────────────────────────────────────────────

    def update_status_bar(self):
        """Synchronise the top-right label with the current GameState."""
        stats = self.game_state.get_status()
        self.status_lbl.config(
            text=(
                f"Remaining: {stats['remaining']}  |  "
                f"Mistakes: {stats['mistakes']}/{GameState.MAX_MISTAKES}  |  "
                f"Score: {stats['score']}"
            )
        )

    # ── REVEAL ───────────────────────────────────────────────────────

    def reveal_differences(self):
        """
        Mark every undiscovered difference with a blue circle on both
        canvases and update the status bar (Remaining → 0).
        Does nothing if no image is loaded or the level is already complete.
        """
        if not self.game_state.regions:
            return

        if self.game_state.is_level_complete():
            return

        revealed = self.game_state.reveal_all()   # clears unfound_regions

        for region in revealed:
            rx, ry, rw, rh = region
            self.draw_circle(rx + rw / 2, ry + rh / 2, "blue")

        self.update_status_bar()   # Remaining now shows 0
        self.bottom_lbl.config(
            text="All differences revealed. Load a new image to play again.",
            fg="#1a6ab5"
        )


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = GameWindow()
    app.mainloop()