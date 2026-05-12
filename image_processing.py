"""
HIT137 Assignment 3: Image Processing Engine
Uses OpenCV for all image manipulation.
"""

import cv2
import numpy as np
import random
import os
import glob


class ImageProcessor:
    """
    Responsible for:
      - Loading images from a local folder (JPG, PNG, BMP)
      - Scaling images to fit a target display size
      - Cloning the original and introducing exactly 5 non-overlapping differences
      - Randomising both type AND position on every load
    """

    NUM_DIFFERENCES   = 5
    MIN_REGION_SIZE   = 50    # minimum region side in pixels
    MAX_REGION_SIZE   = 90    # maximum region side in pixels
    REGION_PADDING    = 15    # minimum gap between regions in pixels
    MAX_PLACE_TRIES   = 2000  # safety cap for placement loop

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
        folder_path     : folder that contains the original image(s)
        display_width   : target canvas width  (for aspect-ratio scaling)
        display_height  : target canvas height (for aspect-ratio scaling)
        """
        self.folder_path     = folder_path
        self.display_width   = display_width
        self.display_height  = display_height

        self.original_path   = None   # path of the source file on disk
        self.modified_path   = None   # path where modified copy is saved

        # Raw full-resolution arrays (BGR, uint8)
        self.original        = None
        self.modified        = None

        # Scaled arrays — what the GUI actually displays
        self.original_display  = None
        self.modified_display  = None

        # scale factor applied during display-scaling (needed to map
        # click coordinates back to full-resolution space)
        self.scale_factor    = 1.0

        # list of (x, y, w, h, alteration_type) in DISPLAY coordinates
        self.regions         = []

        self._ensure_folder()

    # FOLDER MANAGEMENT

    def _ensure_folder(self):
        """Create the working folder if it does not already exist."""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"Created folder: '{self.folder_path}/'")
            print(f"   ➜  Place a JPG / PNG / BMP image inside and run again.")
        else:
            print(f"Using folder: '{self.folder_path}/'")

    def _find_original_image(self) -> str:
        """
        Scan the folder for a source image.
        Files containing '_modified' in their name are ignored so re-runs
        always pick up the original, never the previously generated copy.
        """
        found = []
        for pattern in self.SUPPORTED_FORMATS:
            found.extend(glob.glob(os.path.join(self.folder_path, pattern)))

        originals = [f for f in found if "_modified" not in os.path.basename(f)]

        if not originals:
            raise FileNotFoundError(
                f"\n No original image found in '{self.folder_path}/'.\n"
                f"    Add a JPG, PNG, or BMP image and try again."
            )

        if len(originals) > 1:
            print(f"⚠️   Multiple source images found — using: "
                  f"{os.path.basename(originals[0])}")

        return originals[0]

    def _build_modified_path(self) -> str:
        """
        Build save path for the modified image.
        """
        base = os.path.basename(self.original_path)
        name, ext = os.path.splitext(base)
        return os.path.join(self.folder_path, f"{name}_modified{ext}")

    # LOADING & SCALING 

    def load_image(self) -> bool:
        """
        Auto-detect and load the original image from the folder.
        Supports JPG, JPEG, PNG, BMP.
        """
        self.original_path = self._find_original_image()
        print(f"Loading: {os.path.basename(self.original_path)}")

        img = cv2.imread(self.original_path)
        if img is None:
            raise ValueError(
                f"OpenCV could not read '{self.original_path}'.\n"
                f"The file may be corrupt or an unsupported encoding."
            )

        self.original = img
        h, w = img.shape[:2]
        print(f"Loaded — original size: {w}×{h} px")
        return True

    def load_image_from_path(self, file_path: str) -> tuple:
        """
        Load any image directly from a given file path.
        Used when the user picks a file via Tkinter file dialog.
        Copies the file into images/ folder, then runs the full pipeline.
        Returns (original_display, modified_display, regions)
        """
        import shutil

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Copy the user-selected file into the images folder
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.folder_path, filename)

        if file_path != dest_path:
            shutil.copy2(file_path, dest_path)
            print(f"  Copied '{filename}' → '{self.folder_path}/'")

        # Set as the original and run the pipeline
        self.original_path = dest_path
        img = cv2.imread(dest_path)
        if img is None:
            raise ValueError(f"Could not read image: {dest_path}")

        self.original = img
        h, w = img.shape[:2]
        print(f"  Loaded — size: {w}×{h} px")

        self.original_display = self.scale_to_display(self.original)
        self.clone_image()
        self.generate_differences()
        self.modified_display = self.scale_to_display(self.modified)
        self.save_modified_image()

        return self.original_display, self.modified_display, self.regions

    def scale_to_display(self, image: np.ndarray) -> np.ndarray:
        """
        Scale image to fit within (display_width × display_height)
        while preserving the original aspect ratio.
        Uses INTER_AREA for shrinking (best quality) and
        INTER_LINEAR for enlarging.
        """
        h, w = image.shape[:2]
        scale = min(self.display_width / w, self.display_height / h)
        self.scale_factor = scale

        new_w = int(w * scale)
        new_h = int(h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)

        print(f"Scaled to: {new_w}×{new_h} px  (factor={scale:.3f})")
        return scaled

    # CLONE & DIFFERENCE GENERATION

    def clone_image(self) -> np.ndarray:
        """Create a pixel-perfect deep copy of the original."""
        if self.original is None:
            raise ValueError("Call load_image() first.")
        self.modified = np.copy(self.original)
        return self.modified

    def generate_differences(self) -> list:
        """
        Place exactly NUM_DIFFERENCES (5) non-overlapping altered regions
        on the modified image.

        Both the alteration TYPE and POSITION are chosen randomly on
        every call — satisfies marking criterion:
        "Type and position randomised on every load".

        Returns list of (x, y, w, h, alteration_type)
        in DISPLAY-SCALED coordinates (ready for the GUI canvas).
        """
        if self.modified is None:
            raise ValueError("Call clone_image() first.")

        self.regions = []
        img_h, img_w = self.modified.shape[:2]
        attempts = 0

        # Shuffle alteration list so we never get all the same type
        alteration_pool = self.ALTERATION_TYPES.copy()
        random.shuffle(alteration_pool)

        while len(self.regions) < self.NUM_DIFFERENCES:
            if attempts >= self.MAX_PLACE_TRIES:
                raise RuntimeError(
                    "Could not place 5 non-overlapping regions after "
                    f"{self.MAX_PLACE_TRIES} attempts.\n"
                    " Use an image that is at least 300×300 px."
                )

            w = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            h = random.randint(self.MIN_REGION_SIZE, self.MAX_REGION_SIZE)
            x = random.randint(0, img_w - w - 1)
            y = random.randint(0, img_h - h - 1)

            if not self._has_overlap((x, y, w, h)):
                # Cycle through types so all 4 types appear across a game
                alteration = alteration_pool[len(self.regions) % len(alteration_pool)]
                self._apply_alteration((x, y, w, h), alteration)
                self.regions.append((x, y, w, h, alteration))

            attempts += 1

        print(f"Generated {self.NUM_DIFFERENCES} differences "
              f"(attempts used: {attempts}):")
        for i, (x, y, w, h, alt) in enumerate(self.regions):
            print(f"Region {i+1}: pos=({x},{y})  size={w}×{h}  type={alt}")

        # Convert regions to display-scaled coordinates for the GUI
        self.regions = self._scale_regions(self.regions, self.scale_factor)
        return self.regions

    def _scale_regions(self, regions: list, scale: float) -> list:
        """Convert region coordinates from full-res to display-scaled space."""
        scaled = []
        for (x, y, w, h, alt) in regions:
            scaled.append((
                int(x * scale),
                int(y * scale),
                int(w * scale),
                int(h * scale),
                alt
            ))
        return scaled

    # FOUR ALTERATION TYPES

    def apply_colour_shift(self, region: tuple):
        """
        Shifts the hue of a region by 40–80° in HSV space using OpenCV.
        Auto-detects greyscale regions: if saturation is too low for
        hue shift to be visible, falls back to cv2.bitwise_not which
        inverts intensity (works on white AND black backgrounds).
        """
        x, y, w, h = region
        roi = self.modified[y:y+h, x:x+w]

        # Check saturation — if low, hue shift is invisible (greyscale region)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))

        if mean_saturation < 15:
            # Greyscale region — invert intensity using OpenCV
            # Works on both white (→ dark) and black (→ light) backgrounds
            inverted = cv2.bitwise_not(roi)
            # Blend gently so it's "subtle" not glaring (OpenCV addWeighted)
            blended = cv2.addWeighted(roi, 0.35, inverted, 0.65, 0)
            self.modified[y:y+h, x:x+w] = blended
        else:
            # Colour region — do the standard hue shift via HSV
            hsv = hsv.astype(np.int16)
            shift = random.randint(40, 80)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            self.modified[y:y+h, x:x+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_brightness_patch(self, region: tuple):
        """
        Subtly brightens or darkens a region using OpenCV.
        Uses cv2.convertScaleAbs() which is OpenCV's built-in
        brightness/contrast operation (beta = brightness offset).
        Works on greyscale and colour images alike.
        """
        x, y, w, h = region
        roi = self.modified[y:y+h, x:x+w]

        direction = random.choice([1, -1])
        amount    = random.randint(40, 70)
        beta      = direction * amount

        # cv2.convertScaleAbs handles clamping internally
        brightened = cv2.convertScaleAbs(roi, alpha=1.0, beta=beta)
        self.modified[y:y+h, x:x+w] = brightened

    def apply_edge_overlay(self, region: tuple):
        """
        Uses OpenCV's Canny edge detector + addWeighted to subtly
        embed detected edges into the region.
        Works on ANY image — colour, greyscale, light or dark backgrounds —
        because Canny detects intensity changes, not colour.
        Replaces channel_swap (which fails on greyscale images).
        """
        x, y, w, h = region
        roi = self.modified[y:y+h, x:x+w]

        # Step 1: Canny edge detection (OpenCV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Step 2: Dilate edges so they're more visible (OpenCV morphology)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Step 3: Convert to 3-channel and blend with original (OpenCV)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(roi, 0.75, edges_bgr, 0.45, 0)

        self.modified[y:y+h, x:x+w] = blended

    def apply_gaussian_blur(self, region: tuple):
        """
        Applies a Gaussian blur using OpenCV (kernel 21×21, sigma 8).
        Combined with a tiny brightness nudge so the change remains
        visible even on flat-coloured regions where blur alone produces
        identical pixels.
        Creates a softly out-of-focus and slightly shaded patch.
        """
        x, y, w, h = region
        roi = self.modified[y:y+h, x:x+w]

        # Step 1: Gaussian blur (OpenCV)
        blurred = cv2.GaussianBlur(roi, (21, 21), sigmaX=8)

        # Step 2: Slight brightness nudge so flat regions still differ
        # cv2.convertScaleAbs is OpenCV's brightness adjustment
        direction = random.choice([1, -1])
        blurred = cv2.convertScaleAbs(blurred, alpha=1.0, beta=direction * 15)

        self.modified[y:y+h, x:x+w] = blurred

    # SAVE

    def save_modified_image(self) -> str:
        """
        Save the modified image back into the same folder as the original.
        The filename gets a '_modified' suffix so _find_original_image()
        never mistakes it for a source image on the next run.
        """
        if self.modified is None:
            raise ValueError("No modified image to save.")

        self.modified_path = self._build_modified_path()
        ok = cv2.imwrite(self.modified_path, self.modified)

        if not ok:
            raise IOError(f"cv2.imwrite failed → '{self.modified_path}'")

        size_kb = os.path.getsize(self.modified_path) / 1024
        print(f"Saved modified image: "
              f"{os.path.basename(self.modified_path)}  ({size_kb:.1f} KB)")
        return self.modified_path

    # FULL PIPELINE  (one-call convenience)

    def process(self) -> tuple:
        """
        Full pipeline:
          load → scale (display copies) → clone → generate differences → save

        Returns
        -------
        original_display  : scaled numpy array (BGR) for left canvas
        modified_display  : scaled numpy array (BGR) for right canvas
        regions           : list of (x, y, w, h, alteration_type)
                            in display-scaled coordinates
        """
        self.load_image()

        # Build display-scaled versions (aspect-ratio preserved)
        self.original_display = self.scale_to_display(self.original)

        self.clone_image()
        self.generate_differences()     # regions stored in display coords

        # Build scaled modified AFTER alterations applied to full-res copy
        self.modified_display = self.scale_to_display(self.modified)

        self.save_modified_image()

        return self.original_display, self.modified_display, self.regions

    # HELPERS

    def _apply_alteration(self, region: tuple, alteration_type: str):
        """Dispatch to the correct alteration method."""
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
        Returns True if new_region overlaps (or is too close to) any
        already-placed region.  REGION_PADDING enforces a minimum gap
        so differences remain visually distinct.
        """
        P = self.REGION_PADDING
        x1, y1, w1, h1 = new_region

        for (x2, y2, w2, h2, _) in self.regions:
            no_overlap = (
                x1 + w1 + P <= x2 or   # new is left of existing
                x2 + w2 + P <= x1 or   # new is right of existing
                y1 + h1 + P <= y2 or   # new is above existing
                y2 + h2 + P <= y1      # new is below existing
            )
            if not no_overlap:
                return True     # overlap detected

        return False

    def get_region_centres(self) -> list:
        """
        Return (cx, cy) centre of each region in display coordinates.
        Used by Asim's GameState for proximity-based click detection.
        """
        return [
            (x + w // 2, y + h // 2)
            for (x, y, w, h, _) in self.regions
        ]

    @staticmethod
    def to_tk_image(numpy_image):
        """
        Convert an OpenCV BGR numpy array to a Tkinter-displayable
        PhotoImage. 
        ─────────────────────────────────
        Tkinter does NOT keep a strong reference to PhotoImage objects.
        If the returned image is not stored as an attribute of a
        long-lived object (e.g. self.orig_tk = processor.to_tk_image(...)),
        Python's garbage collector will destroy it immediately and
        the canvas will appear blank — even though circles drawn AFTER
        will still be visible.

        Correct usage:
            self.orig_tk = ImageProcessor.to_tk_image(orig_disp)
            self.canvas.create_image(0, 0, anchor='nw', image=self.orig_tk)
        """
        from PIL import Image, ImageTk
        rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil_img)

    def list_folder_contents(self):
        """Print a summary of all image files in the working folder."""
        print(f"\nContents of '{self.folder_path}/':")
        all_files = []
        for pattern in self.SUPPORTED_FORMATS:
            all_files.extend(
                glob.glob(os.path.join(self.folder_path, pattern))
            )

        if not all_files:
            print("(empty — no images found)")
            return

        for f in sorted(all_files):
            tag  = " ← modified" if "_modified" in os.path.basename(f) else " ← original"
            size = os.path.getsize(f) / 1024
            print(f"    {os.path.basename(f):<45} {size:>7.1f} KB{tag}")

    def __repr__(self):
        return (
            f"ImageProcessor("
            f"folder='{self.folder_path}', "
            f"loaded={self.original is not None}, "
            f"regions={len(self.regions)}, "
            f"scale={self.scale_factor:.3f})"
        )


# DEMO — also produces a visual debug output you can inspect

def run_demo():
    """
    Run the full pipeline on whatever image is in images/
    and save a side-by-side debug PNG showing both images with
    labelled difference boxes.
    """

    FOLDER = "images"
    processor = ImageProcessor(folder_path=FOLDER, display_width=600, display_height=500)
    processor.list_folder_contents()

    try:
        orig_disp, mod_disp, regions = processor.process()
    except FileNotFoundError as e:
        print(e)
        return

    processor.list_folder_contents()
    print(f"\n  {processor}")
    print(f"  Region centres (for GameState): {processor.get_region_centres()}")

    # ── Save side-by-side debug image ──
    debug_orig = orig_disp.copy()
    debug_mod  = mod_disp.copy()
    COLOURS = {
        "colour_shift"    : (0,   255, 255),  # yellow
        "brightness_patch": (255, 165,   0),  # orange
        "edge_overlay"    : (0,   255,   0),  # green
        "gaussian_blur"   : (255,   0, 255),  # magenta
    }

    for i, (x, y, w, h, alt) in enumerate(regions):
        colour = COLOURS.get(alt, (0, 0, 255))
        label  = f"{i+1}:{alt[:4]}"
        for canvas in [debug_orig, debug_mod]:
            cv2.rectangle(canvas, (x, y), (x+w, y+h), colour, 2)
            cv2.putText(canvas, label, (x, max(y-4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                        cv2.LINE_AA)

    separator = np.full((debug_orig.shape[0], 6, 3), 80, dtype=np.uint8)
    side_by_side = np.hstack([debug_orig, separator, debug_mod])

    out_dir  = os.path.join(FOLDER, "debug")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "debug_side_by_side.png")
    cv2.imwrite(out_path, side_by_side)
    print(f"\n  Debug image saved → {out_path}")
    print("       (left = original with labels, right = modified with labels)")


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    run_demo()