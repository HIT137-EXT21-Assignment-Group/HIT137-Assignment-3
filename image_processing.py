"""
HIT137 Assignment 3 — Ishan Panta: Image Processing Engine
"""

import cv2
import numpy as np
import os
import glob
import random


class ImageProcessor:
    """
    Handles image loading from a local folder.
    folder creation, image detection, loading, and saving.
    """

    SUPPORTED_FORMATS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, folder_path: str = "images",
                 display_width: int = 600, display_height: int = 500):
        self.folder_path    = folder_path
        self.display_width  = display_width
        self.display_height = display_height

        self.original_path  = None
        self.modified_path  = None
        self.original       = None
        self.modified       = None
        self.scale_factor   = 1.0
        self.regions        = []

        self._ensure_folder()

    def _ensure_folder(self):
        """Create the working folder if it does not already exist."""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print(f"📁 Created folder: '{self.folder_path}/'")
            print(f"   ➜  Place a JPG / PNG / BMP image inside and run again.")
        else:
            print(f"📁 Using folder: '{self.folder_path}/'")

    def _find_original_image(self) -> str:
        """
        Scan the folder for a source image.
        Logic:
        1. Collect all images that are not '_modified' files
        2. From those, find which ones do NOT already have a modified version
        3. If multiple unprocessed images exist, pick one randomly
        4. If all images already have a modified version, pick randomly from all originals
        """
        # Collect all image files in the folder
        found = []
        for pattern in self.SUPPORTED_FORMATS:
            found.extend(glob.glob(os.path.join(self.folder_path, pattern)))

        # Separate originals from modified files
        originals = [f for f in found if "_modified" not in os.path.basename(f)]

        if not originals:
            raise FileNotFoundError(
                f"\nNo original image found in '{self.folder_path}/'.\n"
                f"    Add a JPG, PNG, or BMP image and try again."
            )

        # Build the expected modified filename for each original
        # e.g. photo.jpg → photo_modified.jpg
        def modified_exists(original_path: str) -> bool:
            name, ext = os.path.splitext(os.path.basename(original_path))
            modified_name = f"{name}_modified{ext}"
            modified_path = os.path.join(self.folder_path, modified_name)
            return os.path.exists(modified_path)

        # Filter to only images that have NO modified version yet
        unprocessed = [f for f in originals if not modified_exists(f)]

        if unprocessed:
            # Pick randomly from images that haven't been processed yet
            chosen = random.choice(unprocessed)
            print(f"Found {len(unprocessed)} unprocessed image(s) — "
                f"randomly chose: {os.path.basename(chosen)}")
        else:
            # All originals already have a modified version — pick any randomly
            chosen = random.choice(originals)
            print(f"All {len(originals)} image(s) already have a modified version.")
            print(f"Randomly chose: {os.path.basename(chosen)} to reprocess.")

        return chosen

    def _build_modified_path(self) -> str:
        """
        Build save path for the modified image.
        Example: images/photo.jpg → images/photo_modified.jpg
        """
        base = os.path.basename(self.original_path)
        name, ext = os.path.splitext(base)
        return os.path.join(self.folder_path, f"{name}_modified{ext}")

    # LOAD & SAVE

    def load_image(self) -> bool:
        """Load the original image from the folder. Supports JPG, PNG, BMP."""
        self.original_path = self._find_original_image()
        print(f" Loading: {os.path.basename(self.original_path)}")

        img = cv2.imread(self.original_path)
        if img is None:
            raise ValueError(
                f"OpenCV could not read '{self.original_path}'.\n"
                f"File may be corrupt or unsupported encoding."
            )

        self.original = img
        h, w = img.shape[:2]
        print(f"Loaded — size: {w}×{h} px")
        return True

    def save_modified_image(self) -> str:
        """Save the modified image back into the same folder as the original."""
        if self.modified is None:
            raise ValueError("No modified image to save.")

        self.modified_path = self._build_modified_path()
        ok = cv2.imwrite(self.modified_path, self.modified)

        if not ok:
            raise IOError(f"cv2.imwrite failed → '{self.modified_path}'")

        size_kb = os.path.getsize(self.modified_path) / 1024
        print(f"Saved: {os.path.basename(self.modified_path)} ({size_kb:.1f} KB)")
        return self.modified_path

    def list_folder_contents(self):
        """Print all image files in the working folder."""
        print(f"\nContents of '{self.folder_path}/':")
        all_files = []
        for pattern in self.SUPPORTED_FORMATS:
            all_files.extend(glob.glob(os.path.join(self.folder_path, pattern)))

        if not all_files:
            print("    (empty — no images found)")
            return

        for f in sorted(all_files):
            tag  = " ← modified" if "_modified" in os.path.basename(f) else " ← original"
            size = os.path.getsize(f) / 1024
            print(f"    {os.path.basename(f):<45} {size:>7.1f} KB{tag}")

    def __repr__(self):
        return (f"ImageProcessor(folder='{self.folder_path}', "
                f"loaded={self.original is not None})")

if __name__ == "__main__":

    FOLDER = "images"

    processor = ImageProcessor(folder_path=FOLDER)
    processor.list_folder_contents()
    processor.load_image()

    # Simulate a modified image (just a copy for now)
    processor.modified = np.copy(processor.original)
    processor.save_modified_image()

    processor.list_folder_contents()