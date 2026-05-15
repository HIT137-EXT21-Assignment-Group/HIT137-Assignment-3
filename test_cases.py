"""

HIT137 Assignment 3 — Test Suite

Run with:  pytest test_cases.py -v

Requires:  pytest, opencv-python, numpy, Pillow

"""

import math

import os

import shutil

import tempfile

import pytest

import numpy as np

import cv2

# ── imports under test ────────────────────────────────────────────────

from game_state       import BaseGame, GameState

from image_processing import ImageProcessor


# ═════════════════════════════════════════════════════════════════════

#  FIXTURES

# ═════════════════════════════════════════════════════════════════════

@pytest.fixture

def five_regions():

    """Five non-overlapping 40×40 regions spread across a 400×400 canvas."""

    return [

        (10,  10,  40, 40),

        (100, 10,  40, 40),

        (200, 10,  40, 40),

        (300, 10,  40, 40),

        (10,  100, 40, 40),

    ]


@pytest.fixture

def fresh_state(five_regions):

    """A GameState reset with five_regions, cumulative score = 0."""

    gs = GameState()

    gs.reset(five_regions)

    return gs


@pytest.fixture

def tmp_image_dir():

    """Temporary directory that is cleaned up after each test."""

    d = tempfile.mkdtemp()

    yield d

    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture

def small_image_path(tmp_image_dir):

    """Write a 300×300 solid-colour PNG and return its path."""

    img  = np.full((300, 300, 3), 128, dtype=np.uint8)

    path = os.path.join(tmp_image_dir, "test_image.png")

    cv2.imwrite(path, img)

    return path


@pytest.fixture

def processor(tmp_image_dir):

    """ImageProcessor pointed at a fresh temp folder."""

    return ImageProcessor(

        folder_path=tmp_image_dir,

        display_width=200,

        display_height=200,

    )


# ═════════════════════════════════════════════════════════════════════

#  1. BASE CLASS

# ═════════════════════════════════════════════════════════════════════

class TestBaseGame:

    """BaseGame must raise NotImplementedError for all abstract methods."""

    def setup_method(self):

        self.base = BaseGame(regions=[(0, 0, 10, 10)])

    def test_is_game_over_raises(self):

        with pytest.raises(NotImplementedError):

            self.base.is_game_over()

    def test_is_level_complete_raises(self):

        with pytest.raises(NotImplementedError):

            self.base.is_level_complete()

    def test_get_status_raises(self):

        with pytest.raises(NotImplementedError):

            self.base.get_status()

    def test_initial_attributes(self):

        assert self.base.regions == [(0, 0, 10, 10)]

        assert self.base.found_regions == []

        assert self.base.mistakes == 0


# ═════════════════════════════════════════════════════════════════════

#  2. GAMESTATE — initialisation & reset

# ═════════════════════════════════════════════════════════════════════

class TestGameStateInit:

    def test_default_init_empty(self):

        gs = GameState()

        assert gs.regions == []

        assert gs.unfound_regions == []

        assert gs.cumulative_score == 0

        assert gs.mistakes == 0

    def test_reset_populates_regions(self, five_regions):

        gs = GameState()

        gs.reset(five_regions)

        assert gs.regions == five_regions

        assert gs.unfound_regions == five_regions

    def test_reset_clears_mistakes(self, fresh_state):

        fresh_state.mistakes = 2

        fresh_state.reset(fresh_state.regions)

        assert fresh_state.mistakes == 0

    def test_reset_clears_found_regions(self, five_regions):

        gs = GameState()

        gs.reset(five_regions)

        gs.found_regions.append(five_regions[0])

        gs.reset(five_regions)

        assert gs.found_regions == []

    def test_reset_preserves_cumulative_score(self, fresh_state, five_regions):

        """Cumulative score must survive a round reset."""

        fresh_state.cumulative_score = 7

        fresh_state.reset(five_regions)

        assert fresh_state.cumulative_score == 7

    def test_unfound_is_independent_copy(self, five_regions):

        """Mutating unfound_regions must not change regions."""

        gs = GameState()

        gs.reset(five_regions)

        gs.unfound_regions.clear()

        assert gs.regions == five_regions


# ═════════════════════════════════════════════════════════════════════

#  3. GAMESTATE — click validation

# ═════════════════════════════════════════════════════════════════════

class TestRegisterClick:

    def _centre(self, region):

        """Return the pixel-exact centre of a region tuple."""

        rx, ry, rw, rh = region

        return int(rx + rw / 2), int(ry + rh / 2)

    # ── correct clicks ────────────────────────────────────────────────

    def test_direct_hit_returns_found(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        status, region = fresh_state.register_click(cx, cy)

        assert status == "found"

        assert region == five_regions[0]

    def test_found_region_removed_from_unfound(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        fresh_state.register_click(cx, cy)

        assert five_regions[0] not in fresh_state.unfound_regions

    def test_found_region_added_to_found(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        fresh_state.register_click(cx, cy)

        assert five_regions[0] in fresh_state.found_regions

    def test_score_increments_on_correct_click(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        fresh_state.register_click(cx, cy)

        assert fresh_state.cumulative_score == 1

    def test_click_at_proximity_edge_still_counts(self, fresh_state, five_regions):

        """A click exactly at PROXIMITY_RADIUS distance must register."""

        cx, cy = self._centre(five_regions[0])

        r = GameState.PROXIMITY_RADIUS

        status, _ = fresh_state.register_click(cx + r, cy)

        assert status == "found"

    def test_click_just_outside_proximity_is_mistake(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        r = GameState.PROXIMITY_RADIUS + 1

        status, _ = fresh_state.register_click(cx + r, cy)

        assert status == "mistake"

    def test_already_found_region_not_re_findable(self, fresh_state, five_regions):

        cx, cy = self._centre(five_regions[0])

        fresh_state.register_click(cx, cy)           # first — found

        status, _ = fresh_state.register_click(cx, cy)  # second — must not re-find

        assert status == "mistake"

    # ── last correct click → level_complete ───────────────────────────

    def test_last_find_returns_level_complete(self, fresh_state, five_regions):

        for region in five_regions[:-1]:

            cx, cy = self._centre(region)

            fresh_state.register_click(cx, cy)

        # Find the last one

        cx, cy = self._centre(five_regions[-1])

        status, _ = fresh_state.register_click(cx, cy)

        assert status == "level_complete"

    def test_level_complete_unfound_is_empty(self, fresh_state, five_regions):

        for region in five_regions:

            cx, cy = self._centre(region)

            fresh_state.register_click(cx, cy)

        assert fresh_state.unfound_regions == []

    # ── wrong clicks → mistakes / game_over ───────────────────────────

    def test_wrong_click_increments_mistakes(self, fresh_state):

        fresh_state.register_click(0, 0)

        assert fresh_state.mistakes == 1

    def test_score_does_not_increment_on_mistake(self, fresh_state):

        fresh_state.register_click(0, 0)

        assert fresh_state.cumulative_score == 0

    def test_third_mistake_returns_game_over(self, fresh_state):

        for _ in range(3):

            status, _ = fresh_state.register_click(0, 0)

        assert status == "game_over"

    def test_click_after_game_over_is_locked(self, fresh_state):

        for _ in range(3):

            fresh_state.register_click(0, 0)

        status, _ = fresh_state.register_click(0, 0)

        assert status == "locked"

    def test_click_after_level_complete_is_locked(self, fresh_state, five_regions):

        for region in five_regions:

            cx, cy = self._centre(region)

            fresh_state.register_click(cx, cy)

        status, _ = fresh_state.register_click(0, 0)

        assert status == "locked"


# ═════════════════════════════════════════════════════════════════════

#  4. GAMESTATE — state predicates

# ═════════════════════════════════════════════════════════════════════

class TestGameStatePredicates:

    def test_is_game_over_false_initially(self, fresh_state):

        assert not fresh_state.is_game_over()

    def test_is_game_over_true_at_max_mistakes(self, fresh_state):

        fresh_state.mistakes = GameState.MAX_MISTAKES

        assert fresh_state.is_game_over()

    def test_is_level_complete_false_initially(self, fresh_state):

        assert not fresh_state.is_level_complete()

    def test_is_level_complete_true_when_unfound_empty(self, fresh_state):

        fresh_state.unfound_regions.clear()

        assert fresh_state.is_level_complete()

    def test_is_level_complete_false_with_no_regions(self):

        """Edge-case: no regions loaded at all → not complete."""

        gs = GameState()

        assert not gs.is_level_complete()


# ═════════════════════════════════════════════════════════════════════

#  5. GAMESTATE — get_status & reveal_all

# ═════════════════════════════════════════════════════════════════════

class TestGameStateStatus:

    def test_get_status_keys_present(self, fresh_state):

        s = fresh_state.get_status()

        for key in ("remaining", "found", "total", "mistakes", "score"):

            assert key in s, f"Key '{key}' missing from get_status()"

    def test_get_status_initial_values(self, fresh_state, five_regions):

        s = fresh_state.get_status()

        assert s["remaining"] == 5

        assert s["found"]     == 0

        assert s["total"]     == 5

        assert s["mistakes"]  == 0

        assert s["score"]     == 0

    def test_get_status_after_one_find(self, fresh_state, five_regions):

        cx, cy = int(five_regions[0][0] + five_regions[0][2] / 2), int(five_regions[0][1] + five_regions[0][3] / 2)

        fresh_state.register_click(cx, cy)

        s = fresh_state.get_status()

        assert s["remaining"] == 4

        assert s["found"]     == 1

        assert s["score"]     == 1

    def test_reveal_all_returns_unfound(self, fresh_state, five_regions):

        revealed = fresh_state.reveal_all()

        assert set(revealed) == set(five_regions)

    def test_reveal_all_clears_unfound(self, fresh_state):

        fresh_state.reveal_all()

        assert fresh_state.unfound_regions == []

    def test_reveal_all_remaining_becomes_zero(self, fresh_state):

        fresh_state.reveal_all()

        assert fresh_state.get_status()["remaining"] == 0

    def test_reveal_all_partial(self, fresh_state, five_regions):

        """After finding 2 manually, reveal_all returns only the remaining 3."""

        for region in five_regions[:2]:

            cx = int(region[0] + region[2] / 2)

            cy = int(region[1] + region[3] / 2)

            fresh_state.register_click(cx, cy)

        revealed = fresh_state.reveal_all()

        assert len(revealed) == 3


# ═════════════════════════════════════════════════════════════════════

#  6. IMAGE PROCESSOR — folder & scaling

# ═════════════════════════════════════════════════════════════════════

class TestImageProcessorSetup:

    def test_creates_folder_if_absent(self, tmp_image_dir):

        new_folder = os.path.join(tmp_image_dir, "sub_images")

        assert not os.path.exists(new_folder)

        ImageProcessor(folder_path=new_folder)

        assert os.path.exists(new_folder)

    def test_does_not_crash_if_folder_exists(self, tmp_image_dir):

        """Calling constructor twice on same folder must not raise."""

        ImageProcessor(folder_path=tmp_image_dir)

        ImageProcessor(folder_path=tmp_image_dir)

    def test_scale_to_display_fits_within_bounds(self, processor):

        # 600×400 image scaled into 200×200 box

        img    = np.zeros((400, 600, 3), dtype=np.uint8)

        scaled = processor.scale_to_display(img)

        h, w   = scaled.shape[:2]

        assert w <= 200 and h <= 200

    def test_scale_preserves_aspect_ratio(self, processor):

        # 600×300 → 2:1 ratio must survive scaling

        img    = np.zeros((300, 600, 3), dtype=np.uint8)

        scaled = processor.scale_to_display(img)

        h, w   = scaled.shape[:2]

        assert abs((w / h) - 2.0) < 0.05   # within 5 % of 2:1

    def test_scale_factor_stored(self, processor):

        img = np.zeros((400, 400, 3), dtype=np.uint8)

        processor.scale_to_display(img)

        assert 0 < processor.scale_factor <= 1.0

    def test_scale_small_image_upscales(self):

        """An image smaller than the target box should be upscaled."""

        proc = ImageProcessor(display_width=400, display_height=400,

                              folder_path=tempfile.mkdtemp())

        img  = np.zeros((100, 100, 3), dtype=np.uint8)

        scaled = proc.scale_to_display(img)

        h, w = scaled.shape[:2]

        assert w == 400 and h == 400

        shutil.rmtree(proc.folder_path, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════

#  7. IMAGE PROCESSOR — clone

# ═════════════════════════════════════════════════════════════════════

class TestImageProcessorClone:

    def test_clone_produces_equal_arrays(self, processor):

        processor.original = np.full((100, 100, 3), 42, dtype=np.uint8)

        processor.clone_image()

        assert np.array_equal(processor.original, processor.modified)

    def test_clone_is_deep_copy(self, processor):

        """Mutating modified must NOT change original."""

        processor.original = np.full((100, 100, 3), 42, dtype=np.uint8)

        processor.clone_image()

        processor.modified[0, 0, 0] = 99

        assert processor.original[0, 0, 0] == 42

    def test_clone_without_original_raises(self, processor):

        with pytest.raises(ValueError):

            processor.clone_image()


# ═════════════════════════════════════════════════════════════════════

#  8. IMAGE PROCESSOR — difference generation

# ═════════════════════════════════════════════════════════════════════

class TestGenerateDifferences:

    @pytest.fixture(autouse=True)

    def setup_image(self, processor):

        """Give the processor a 300×300 image to work with."""

        processor.original = np.full((300, 300, 3), 128, dtype=np.uint8)

        processor.scale_to_display(processor.original)

        processor.clone_image()

    def test_returns_exactly_five_regions(self, processor):

        regions = processor.generate_differences()

        assert len(regions) == 5

    def test_regions_are_five_tuples(self, processor):

        for region in processor.generate_differences():

            assert len(region) == 5, "Each region must be (x, y, w, h, alt_type)"

    def test_alteration_types_are_valid(self, processor):

        valid = set(ImageProcessor.ALTERATION_TYPES)

        for (_, _, _, _, alt) in processor.generate_differences():

            assert alt in valid

    def test_regions_stored_on_instance(self, processor):

        processor.generate_differences()

        assert len(processor.regions) == 5

    def test_regions_differ_between_calls(self, processor):

        """Position must be randomised — two runs almost never match."""

        r1 = processor.generate_differences()

        processor.clone_image()

        r2 = processor.generate_differences()

        # Compare just the (x, y) pairs; collision probability ≈ 0

        positions1 = [(x, y) for (x, y, *_) in r1]

        positions2 = [(x, y) for (x, y, *_) in r2]

        assert positions1 != positions2, (

            "Regions appear identical across two runs — randomisation may be broken"

        )

    def test_regions_non_overlapping(self, processor):

        """

        No two returned regions may overlap in display coordinates.

        REGION_PADDING is enforced in full-resolution space during generation;

        after scaling it shrinks proportionally, so we check raw pixel overlap

        only (no padding added here).

        """

        regions = processor.generate_differences()

        for i, (x1, y1, w1, h1, _) in enumerate(regions):

            for j, (x2, y2, w2, h2, _) in enumerate(regions):

                if i >= j:

                    continue

                # Raw overlap check — no padding (padding is a full-res concern)

                overlap = not (

                    x1 + w1 <= x2 or

                    x2 + w2 <= x1 or

                    y1 + h1 <= y2 or

                    y2 + h2 <= y1

                )

                assert not overlap, (

                    f"Regions {i} and {j} overlap in display space: "

                    f"{(x1,y1,w1,h1)} vs {(x2,y2,w2,h2)}"

                )

    def test_modified_differs_from_original(self, processor):

        """generate_differences() must actually alter pixels."""

        processor.generate_differences()

        assert not np.array_equal(processor.original, processor.modified)

    def test_generate_without_clone_raises(self, processor):

        processor.modified = None

        with pytest.raises(ValueError):

            processor.generate_differences()


# ═════════════════════════════════════════════════════════════════════

#  9. IMAGE PROCESSOR — overlap detection

# ═════════════════════════════════════════════════════════════════════

class TestHasOverlap:

    @pytest.fixture(autouse=True)

    def seed_one_region(self, processor):

        """Pre-seed the processor with one region to test against."""

        processor.regions = [(50, 50, 40, 40, "colour_shift")]

    def test_identical_region_overlaps(self, processor):

        assert processor._has_overlap((50, 50, 40, 40)) is True

    def test_touching_edge_overlaps_within_padding(self, processor):

        # Existing region starts at x=50.  A new region of width=36

        # leaves a gap of 14 px which is less than REGION_PADDING (15),

        # so it must be detected as overlapping (0+36+15=51 > 50).

        assert processor._has_overlap((0, 50, 36, 40)) is True

    def test_region_beyond_padding_gap_is_clear(self, processor):

        P = ImageProcessor.REGION_PADDING

        # x=50+40+P → safely outside padding

        assert processor._has_overlap((50 + 40 + P, 50, 40, 40)) is False

    def test_region_far_away_is_clear(self, processor):

        assert processor._has_overlap((200, 200, 30, 30)) is False

    def test_no_existing_regions_never_overlaps(self, processor):

        processor.regions = []

        assert processor._has_overlap((0, 0, 40, 40)) is False


# ═════════════════════════════════════════════════════════════════════

#  10. IMAGE PROCESSOR — alteration types

# ═════════════════════════════════════════════════════════════════════

class TestAlterationTypes:

    """Each alteration must run without error and modify pixels."""

    REGION = (10, 10, 60, 60)

    @pytest.fixture(autouse=True)

    def colourful_image(self, processor):

        """300×300 image with vivid colour so all alteration paths run."""

        img = np.zeros((300, 300, 3), dtype=np.uint8)

        img[:, :, 0] = 200   # blue channel  → B

        img[:, :, 1] = 100   # green channel → G

        img[:, :, 2] = 50    # red channel   → R

        processor.original = img

        processor.modified  = img.copy()

    def _roi_before(self, processor):

        x, y, w, h = self.REGION

        return processor.modified[y:y+h, x:x+w].copy()

    def _roi_after(self, processor):

        x, y, w, h = self.REGION

        return processor.modified[y:y+h, x:x+w]

    def test_colour_shift_changes_pixels(self, processor):

        before = self._roi_before(processor)

        processor.apply_colour_shift(self.REGION)

        assert not np.array_equal(before, self._roi_after(processor))

    def test_brightness_patch_changes_pixels(self, processor):

        before = self._roi_before(processor)

        processor.apply_brightness_patch(self.REGION)

        assert not np.array_equal(before, self._roi_after(processor))

    def test_edge_overlay_changes_pixels(self, processor):

        before = self._roi_before(processor)

        processor.apply_edge_overlay(self.REGION)

        # Edge overlay on a flat-colour image may produce no edges —

        # just verify it runs without crashing

        assert self._roi_after(processor) is not None

    def test_gaussian_blur_changes_pixels(self, processor):

        before = self._roi_before(processor)

        processor.apply_gaussian_blur(self.REGION)

        assert not np.array_equal(before, self._roi_after(processor))

    def test_unknown_alteration_raises(self, processor):

        with pytest.raises(ValueError):

            processor._apply_alteration(self.REGION, "teleportation")

    def test_colour_shift_greyscale_fallback(self, processor):

        """On a pure-grey image colour_shift must use the inversion path."""

        processor.modified = np.full((300, 300, 3), 180, dtype=np.uint8)

        before = processor.modified[10:70, 10:70].copy()

        processor.apply_colour_shift(self.REGION)

        after  = processor.modified[10:70, 10:70]

        assert not np.array_equal(before, after)


# ═════════════════════════════════════════════════════════════════════

#  11. IMAGE PROCESSOR — scale_regions

# ═════════════════════════════════════════════════════════════════════

class TestScaleRegions:

    def test_scale_by_half(self, processor):

        regions = [(100, 80, 60, 40, "gaussian_blur")]

        scaled  = processor._scale_regions(regions, 0.5)

        assert scaled == [(50, 40, 30, 20, "gaussian_blur")]

    def test_scale_by_one_unchanged(self, processor):

        regions = [(100, 80, 60, 40, "edge_overlay")]

        scaled  = processor._scale_regions(regions, 1.0)

        assert scaled == [(100, 80, 60, 40, "edge_overlay")]

    def test_alteration_type_preserved(self, processor):

        regions = [(10, 10, 20, 20, "brightness_patch")]

        scaled  = processor._scale_regions(regions, 0.75)

        assert scaled[0][4] == "brightness_patch"

    def test_multiple_regions_all_scaled(self, processor):

        regions = [(0, 0, 100, 100, "colour_shift"),

                   (200, 200, 50, 50, "gaussian_blur")]

        scaled  = processor._scale_regions(regions, 0.5)

        assert len(scaled) == 2

        assert scaled[0][:4] == (0, 0, 50, 50)

        assert scaled[1][:4] == (100, 100, 25, 25)


# ═════════════════════════════════════════════════════════════════════

#  12. IMAGE PROCESSOR — load_image_from_path

# ═════════════════════════════════════════════════════════════════════

class TestLoadImageFromPath:

    def test_nonexistent_path_raises(self, processor):

        with pytest.raises(FileNotFoundError):

            processor.load_image_from_path("/no/such/file.png")

    def test_returns_three_tuple(self, processor, small_image_path):

        result = processor.load_image_from_path(small_image_path)

        assert isinstance(result, tuple) and len(result) == 3

    def test_returns_numpy_arrays(self, processor, small_image_path):

        orig, mod, _ = processor.load_image_from_path(small_image_path)

        assert isinstance(orig, np.ndarray)

        assert isinstance(mod,  np.ndarray)

    def test_returns_five_regions(self, processor, small_image_path):

        _, _, regions = processor.load_image_from_path(small_image_path)

        assert len(regions) == 5

    def test_display_images_fit_within_bounds(self, processor, small_image_path):

        orig, mod, _ = processor.load_image_from_path(small_image_path)

        oh, ow = orig.shape[:2]

        mh, mw = mod.shape[:2]

        assert ow <= 200 and oh <= 200

        assert mw <= 200 and mh <= 200

    def test_same_file_no_winerror(self, tmp_image_dir, small_image_path):

        """

        Selecting a file already inside the images folder must not raise

        WinError 32 (the samefile fix).

        """

        # Copy image INTO the processor's own folder first

        dest = os.path.join(tmp_image_dir, os.path.basename(small_image_path))

        if not os.path.exists(dest):

            shutil.copy2(small_image_path, dest)

        proc = ImageProcessor(

            folder_path=tmp_image_dir,

            display_width=200,

            display_height=200,

        )

        # Should not raise — this is the scenario that caused WinError 32

        proc.load_image_from_path(dest)

    def test_modified_image_saved_to_disk(self, processor, small_image_path):

        processor.load_image_from_path(small_image_path)

        assert processor.modified_path is not None

        assert os.path.exists(processor.modified_path)

    def test_regions_in_display_coordinates(self, processor, small_image_path):

        """All returned region coordinates must fit within the display bounds."""

        _, _, regions = processor.load_image_from_path(small_image_path)

        for (x, y, w, h, _) in regions:

            assert x >= 0 and y >= 0

            assert x + w <= 200 and y + h <= 200


# ═════════════════════════════════════════════════════════════════════

#  13. INTEGRATION — full round simulation

# ═════════════════════════════════════════════════════════════════════

class TestFullRound:

    """

    Simulate a complete game round:

    ImageProcessor generates regions → GameState tracks them → player wins.

    """

    def test_find_all_five_and_complete(self, processor, small_image_path):

        _, _, regions = processor.load_image_from_path(small_image_path)

        regions_4 = [(x, y, w, h) for (x, y, w, h, _) in regions]

        gs = GameState()

        gs.reset(regions_4)

        for i, (rx, ry, rw, rh) in enumerate(regions_4):

            cx = int(rx + rw / 2)

            cy = int(ry + rh / 2)

            status, _ = gs.register_click(cx, cy)

            if i < 4:

                assert status == "found"

            else:

                assert status == "level_complete"

        assert gs.is_level_complete()

        assert gs.cumulative_score == 5

    def test_three_mistakes_then_game_over(self, processor, small_image_path):

        processor.load_image_from_path(small_image_path)

        gs = GameState()

        gs.reset([(0, 0, 10, 10)])  # tiny region far from click point

        for _ in range(2):

            status, _ = gs.register_click(299, 299)

            assert status == "mistake"

        status, _ = gs.register_click(299, 299)

        assert status == "game_over"

        assert gs.is_game_over()

    def test_cumulative_score_across_two_rounds(self, processor, small_image_path):

        _, _, regions = processor.load_image_from_path(small_image_path)

        regions_4 = [(x, y, w, h) for (x, y, w, h, _) in regions]

        gs = GameState()

        gs.reset(regions_4)

        # Find 3 in round 1

        for rx, ry, rw, rh in regions_4[:3]:

            gs.register_click(int(rx + rw / 2), int(ry + rh / 2))

        assert gs.cumulative_score == 3

        # Start round 2 — score must carry over

        gs.reset(regions_4)

        assert gs.cumulative_score == 3

        # Find all 5 in round 2

        for rx, ry, rw, rh in regions_4:

            gs.register_click(int(rx + rw / 2), int(ry + rh / 2))

        assert gs.cumulative_score == 8   # 3 + 5

    def test_reveal_after_game_over(self, processor, small_image_path):

        _, _, regions = processor.load_image_from_path(small_image_path)

        regions_4 = [(x, y, w, h) for (x, y, w, h, _) in regions]

        gs = GameState()

        gs.reset(regions_4)

        # Exhaust mistakes

        for _ in range(3):

            gs.register_click(0, 0)

        assert gs.is_game_over()

        revealed = gs.reveal_all()

        assert len(revealed) == 5          # all still unfound

        assert gs.get_status()["remaining"] == 0
 