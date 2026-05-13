"""
HIT137 Assignment 3 — Image Processing Engine
All image manipulation performed exclusively with OpenCV.
"""

import cv2
import numpy as np
import random
import os
import glob


class ImageProcessor:
    """
    Responsible for:
      - Loading images from disk (JPG, JPEG, PNG, BMP)
      - Scaling to fit any target canvas size (aspect-ratio preserving)
      - Deep-cloning the original and introducing exactly 5 non-overlapping
        programmatic differences, with randomised type AND position every load
      - Saving the modified image back to disk

    Four alteration types satisfy the "3 or more distinct types" criterion:
        colour_shift      — HSV hue rotation (or intensity inversion on grey)
        brightness_patch  — cv2.convertScaleAbs brightness offset
        edge_overlay      — Canny edges blended in with addWeighted
        gaussian_blur     — GaussianBlur + tiny brightness nudge
    """

    NUM_DIFFERENCES = 5
    MIN_REGION_SIZE = 50      # minimum region side in pixels (full-res)
    MAX_REGION_SIZE = 90      # maximum region side in pixels (full-res)
    REGION_PADDING  = 15      # minimum gap between any two regions
    MAX_PLACE_TRIES = 2000    # safety cap for the placement loop

    ALTERATION_TYPES = [
        "colour_shift",
        "brightness_patch",
        "edge_overlay",
        "gaussian_blur",
    ]

    SUPPORTED_FORMATS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, folder_path: str = "images",
                 display_width: int = 600, display_height: int = 500):
        """
        Parameters
        ----------
        folder_path    : working folder (created automatically if absent)
        display_width  : target canvas width  — images are scaled to fit
        display_height : target canvas height — images are scaled to fit
        """
        self.folder_path    = folder_path
        self.display_width  = display_width
        self.display_height = display_height

        self.original_path  = None
        self.modified_path  = None

        self.original         = None   # full-resolution BGR array
        self.modified         = None   # full-resolution modified BGR array

        self.original_display = None   # display-scaled BGR array (for GUI)
        self.modified_display = None   # display-scaled BGR array (for GUI)

        self.scale_factor     = 1.0    # used to map regions → display coords
        self.regions          = []     # list of (x, y, w, h, alteration_type)
                                       # in display-scaled coordinates

        self._ensure_folder()

    # ── FOLDER MANAGEMENT ─────────────────────────────────────────────

    def _ensure_folder(self):
        """Create the working folder if it does not already exist."""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"Created images folder: '{self.folder_path}/'")
        else:
            print(f"Using images folder:   '{self.folder_path}/'")

    def _build_modified_path(self) -> str:
        base = os.path.basename(self.original_path)
        name, ext = os.path.splitext(base)
        return os.path.join(self.folder_path, f"{name}_modified{ext}")

    # ── LOADING ───────────────────────────────────────────────────────

    def load_image_from_path(self, file_path: str) -> tuple:
        """
        Load any image from a file path chosen by the user.

        Copies the file into images/ only when source and destination are
        different files.  Uses os.path.samefile() for a Windows-safe
        comparison (handles case-insensitivity and path normalisation —
        prevents WinError 32 when the selected file is already in images/).

        Runs the full pipeline: scale → clone → generate differences → save.

        Returns
        -------
        (original_display, modified_display, regions)
        where regions is a list of (x, y, w, h, alteration_type)
        in display-scaled coordinates.
        """
        import shutil

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename  = os.path.basename(file_path)
        dest_path = os.path.join(self.folder_path, filename)

        # Windows-safe same-file check prevents copying a file onto itself
        same_file = (
            os.path.exists(dest_path)
            and os.path.samefile(file_path, dest_path)
        )
        if not same_file:
            shutil.copy2(file_path, dest_path)
            print(f"  Copied '{filename}' → '{self.folder_path}/'")

        self.original_path = dest_path
        img = cv2.imread(dest_path)
        if img is None:
            raise ValueError(
                f"OpenCV could not read '{dest_path}'.\n"
                "The file may be corrupt or an unsupported encoding."
            )

        self.original = img
        h, w = img.shape[:2]
        print(f"  Loaded '{filename}' — original size: {w}×{h} px")

        # Full pipeline
        self.original_display = self.scale_to_display(self.original)
        self.clone_image()
        self.generate_differences()           # operates on full-res modified
        self.modified_display = self.scale_to_display(self.modified)
        self.save_modified_image()

        return self.original_display, self.modified_display, self.regions

    # ── SCALING ───────────────────────────────────────────────────────

    def scale_to_display(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to fit within (display_width × display_height)
        while preserving aspect ratio.
        INTER_AREA for downscaling (best quality), INTER_LINEAR for upscaling.
        """
        h, w = image.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        self.scale_factor = scale

        new_w = int(w * scale)
        new_h = int(h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)
        print(f"  Scaled to: {new_w}×{new_h} px  (factor={scale:.3f})")
        return scaled

    # ── CLONE ─────────────────────────────────────────────────────────

    def clone_image(self) -> np.ndarray:
        """Create a pixel-perfect deep copy of the original."""
        if self.original is None:
            raise ValueError("No image loaded — call load_image_from_path() first.")
        self.modified = np.copy(self.original)
        return self.modified

    # ── DIFFERENCE GENERATION ─────────────────────────────────────────

    def generate_differences(self) -> list:
        """
        Place exactly NUM_DIFFERENCES (5) non-overlapping altered regions
        on the FULL-RESOLUTION modified image, then convert their coordinates
        to display-scaled space for the GUI canvas.

        Both alteration TYPE and POSITION are randomised on every call.
        Returns list of (x, y, w, h, alteration_type) in display coordinates.
        """
        if self.modified is None:
            raise ValueError("Call clone_image() first.")

        self.regions = []
        img_h, img_w = self.modified.shape[:2]
        attempts = 0

        # Shuffle so we cycle through all 4 types across the 5 regions
        pool = self.ALTERATION_TYPES.copy()
        random.shuffle(pool)

        while len(self.regions) < self.NUM_DIFFERENCES:
            if attempts >= self.MAX_PLACE_TRIES:
                raise RuntimeError(
                    f"Could not place {self.NUM_DIFFERENCES} non-overlapping "
                    f"regions after {self.MAX_PLACE_TRIES} attempts. "
                    "Use an image at least 300×300 px."
                )

            w = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            h = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            x = random.randint(0, img_w - w - 1)
            y = random.randint(0, img_h - h - 1)

            if not self._has_overlap((x, y, w, h)):
                alt = pool[len(self.regions) % len(pool)]
                self._apply_alteration((x, y, w, h), alt)
                self.regions.append((x, y, w, h, alt))

            attempts += 1

        print(f"  Generated {self.NUM_DIFFERENCES} differences "
              f"({attempts} placement attempts):")
        for i, (x, y, w, h, alt) in enumerate(self.regions):
            print(f"    Region {i+1}: ({x},{y}) {w}×{h}  [{alt}]")

        # Convert to display coordinates before returning to the GUI
        self.regions = self._scale_regions(self.regions, self.scale_factor)
        return self.regions

    def _scale_regions(self, regions: list, scale: float) -> list:
        """Map region coordinates from full-resolution to display space."""
        return [
            (int(x * scale), int(y * scale),
             int(w * scale), int(h * scale), alt)
            for (x, y, w, h, alt) in regions
        ]

    # ── FOUR ALTERATION TYPES ─────────────────────────────────────────

    def apply_colour_shift(self, region: tuple):
        """
        Shifts hue by 40–80° in HSV space (OpenCV).
        Falls back to cv2.bitwise_not intensity inversion for greyscale
        regions where hue shift would be invisible.
        Subtle enough to require careful inspection.
        """
        x, y, w, h = region
        roi = self.modified[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if float(np.mean(hsv[:, :, 1])) < 15:
            # Greyscale — invert intensity, blend gently
            inverted = cv2.bitwise_not(roi)
            blended  = cv2.addWeighted(roi, 0.35, inverted, 0.65, 0)
            self.modified[y:y+h, x:x+w] = blended
        else:
            # Colour — rotate hue
            hsv_int = hsv.astype(np.int16)
            shift   = random.randint(40, 80)
            hsv_int[:, :, 0] = (hsv_int[:, :, 0] + shift) % 180
            hsv_uint = np.clip(hsv_int, 0, 255).astype(np.uint8)
            self.modified[y:y+h, x:x+w] = cv2.cvtColor(hsv_uint,
                                                         cv2.COLOR_HSV2BGR)

    def apply_brightness_patch(self, region: tuple):
        """
        Brightens or darkens a region using cv2.convertScaleAbs (OpenCV).
        Works on both colour and greyscale images; clamping is automatic.
        """
        x, y, w, h = region
        roi       = self.modified[y:y+h, x:x+w]
        direction = random.choice([1, -1])
        beta      = direction * random.randint(40, 70)
        self.modified[y:y+h, x:x+w] = cv2.convertScaleAbs(
            roi, alpha=1.0, beta=beta
        )

    def apply_edge_overlay(self, region: tuple):
        """
        Detects edges with cv2.Canny then blends them into the region
        using cv2.addWeighted.  Works on any image type because Canny
        operates on intensity, not colour.
        """
        x, y, w, h = region
        roi  = self.modified[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edges  = cv2.Canny(gray, threshold1=50, threshold2=150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges  = cv2.dilate(edges, kernel, iterations=1)

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        blended   = cv2.addWeighted(roi, 0.75, edges_bgr, 0.45, 0)
        self.modified[y:y+h, x:x+w] = blended

    def apply_gaussian_blur(self, region: tuple):
        """
        Applies cv2.GaussianBlur (21×21, sigma 8) plus a small brightness
        nudge (cv2.convertScaleAbs) so the patch remains visible even on
        flat-colour regions where blur alone produces identical pixels.
        """
        x, y, w, h = region
        roi     = self.modified[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (21, 21), sigmaX=8)
        direction = random.choice([1, -1])
        blurred = cv2.convertScaleAbs(blurred, alpha=1.0, beta=direction * 15)
        self.modified[y:y+h, x:x+w] = blurred

    # ── SAVE ──────────────────────────────────────────────────────────

    def save_modified_image(self) -> str:
        """
        Save the modified (full-resolution) image with a '_modified' suffix.
        The suffix ensures _find_original_image never mistakes it for a
        source image on the next run.
        """
        if self.modified is None:
            raise ValueError("No modified image to save.")

        self.modified_path = self._build_modified_path()
        if not cv2.imwrite(self.modified_path, self.modified):
            raise IOError(f"cv2.imwrite failed → '{self.modified_path}'")

        size_kb = os.path.getsize(self.modified_path) / 1024
        print(f"  Saved modified image: "
              f"'{os.path.basename(self.modified_path)}'  ({size_kb:.1f} KB)")
        return self.modified_path

    # ── HELPERS ───────────────────────────────────────────────────────

    def _apply_alteration(self, region: tuple, alteration_type: str):
        """Dispatch to the correct alteration method by name."""
        dispatch = {
            "colour_shift"    : self.apply_colour_shift,
            "brightness_patch": self.apply_brightness_patch,
            "edge_overlay"    : self.apply_edge_overlay,
            "gaussian_blur"   : self.apply_gaussian_blur,
        }
        if alteration_type not in dispatch:
            raise ValueError(f"Unknown alteration type: '{alteration_type}'")
        dispatch[alteration_type](region)

    def _has_overlap(self, new_region: tuple) -> bool:
        """
        Returns True if new_region overlaps or is closer than REGION_PADDING
        pixels to any already-placed region.
        """
        P  = self.REGION_PADDING
        x1, y1, w1, h1 = new_region

        for (x2, y2, w2, h2, _) in self.regions:
            no_overlap = (
                x1 + w1 + P <= x2 or
                x2 + w2 + P <= x1 or
                y1 + h1 + P <= y2 or
                y2 + h2 + P <= y1
            )
            if not no_overlap:
                return True
        return False

    def get_region_centres(self) -> list:
        """Return (cx, cy) for each region in display coordinates."""
        return [
            (x + w // 2, y + h // 2)
            for (x, y, w, h, _) in self.regions
        ]

    @staticmethod
    def to_tk_image(numpy_image):
        """
        Convert an OpenCV BGR numpy array to a Tkinter PhotoImage.
        The caller MUST store the returned object as a long-lived attribute
        (e.g. self.photo_left = ...) — if it goes out of scope the canvas
        will go blank even though any circles drawn on top remain visible.
        """
        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def __repr__(self):
        return (
            f"ImageProcessor(folder='{self.folder_path}', "
            f"loaded={self.original is not None}, "
            f"regions={len(self.regions)}, "
            f"scale={self.scale_factor:.3f})"
        )