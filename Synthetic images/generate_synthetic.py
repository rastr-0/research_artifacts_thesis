#!/usr/bin/env python3
"""
generate_synthetic.py
=====================

Makes 6 synthetic "SEM cross-section" images of a thin magnesium wire with a
corrosion layer around it (images 1-5, plus the harder image 6,
"06_hard_realistic", which has its own section further down).  Because we build the images ourselves from simple
formulas, we know the TRUE layer thickness at every angle exactly.  That lets
you test thickness-measurement methods in Fiji/ImageJ against a known answer.

How the geometry is defined (the "ground truth")
------------------------------------------------
Everything is described in polar coordinates around the CENTRE OF THE IMAGE.

    r_in(theta)  = radius of the metal core at angle theta        (micrometres)
    r_out(theta) = outer radius of the corrosion layer             (micrometres)
    t(theta)     = r_out(theta) - r_in(theta) = TRUE THICKNESS     (micrometres)

So the true thickness is measured RADIALLY, i.e. along a straight line from the
image centre outwards.

Angle convention: theta = 0 deg points to the RIGHT (3 o'clock) and the angle
grows COUNTER-CLOCKWISE as you look at the image on screen (90 deg = up,
180 deg = left, 270 deg = down).

The image centre is the exact middle of the image: for a 600 x 600 image that
is the corner point shared by pixels 299 and 300, i.e. x = y = 300 px in
ImageJ coordinates (= 165 um after calibration).
EXCEPTION: image 6 uses the wire centre IMG6_CENTRE_PX = (303.4, 296.8) px,
deliberately a few px away from the image centre.

How the pixels are made
-----------------------
1. We make a 4x finer grid of "sub-pixels" (SUPERSAMPLE = 4).
2. For each sub-pixel we compute its radius r and angle theta and decide if it
   is resin, corrosion layer or metal by comparing r with r_in(theta) and
   r_out(theta).
3. We give each class its grey value (plus texture in the layer).
4. We average each 4x4 block of sub-pixels into one real pixel.  Pixels that
   sit on a boundary therefore get an in-between grey value ("anti-aliasing"),
   just like a real camera/detector would produce, and the edges are not
   shifted by the pixel grid.
5. Then blur (Gaussian, sigma ~1 px) and detector noise are added, and the
   result is rounded to 8-bit (0..255).

The label mask (<name>_labels.tif) is decided at the CENTRE of each real pixel
using the exact formulas: 0 = resin, 1 = corrosion layer, 2 = metal.  It is
the ideal geometry, made before cracks, fragments, texture, noise and blur.

Run it with:   python generate_synthetic.py [output folder]
(default output folder = the folder of this script.  Besides the images it
also writes ground_truth_summary.csv, overview.png and overview_06.png there.)
(needs numpy, scipy, scikit-image, pillow, matplotlib, tifffile)
"""

import csv
import os
import sys

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import measure

# =============================================================================
# PARAMETERS  (change things here)
# =============================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PIXEL_SIZE_UM = 0.55          # micrometres per pixel (real SEM calibration)
IMAGE_SIZE_PX = 600           # image is IMAGE_SIZE_PX x IMAGE_SIZE_PX
SUPERSAMPLE = 4               # sub-pixels per pixel along each axis

METAL_RADIUS_UM = 140.0       # nominal radius of the metal core

# Grey values (0..255), measured on a real sample
GREY_RESIN = 59.0
GREY_METAL = 143.0
GREY_LAYER = 116.0

# Noise / texture for the "realistic" images
NOISE_SD_RESIN = 6.0          # random pixel noise in the resin
NOISE_SD_METAL = 5.0          # random pixel noise in the metal
NOISE_SD_LAYER = 6.0          # random pixel noise in the layer
LAYER_TEXTURE_SD = 24.0       # strength of the blotchy layer texture (before blur)
LAYER_TEXTURE_GRAIN_PX = 1.0  # size of the texture blotches (Gaussian sigma, px)
BLUR_SIGMA_PX = 1.0           # edge blur (Gaussian sigma, px)

JPEG_QUALITY = 75             # quality of the extra .jpg copy of image 5

# Angles at which the truth table (<name>_truth.csv) is written
TRUTH_ANGLE_STEP_DEG = 0.5

# Random seeds (fixed so that every run gives identical images)
SEED_IMAGE_2 = 2              # texture + noise of image 2
SEED_IMAGE_3 = 3              # texture + noise of image 3
SEED_IMAGE_4 = 4              # geometry + texture + noise of image 4
SEED_IMAGE_5 = 5              # geometry + texture + noise of image 5

# Thickness model for images 4 and 5 (the "uniform" corrosion part)
UNIFORM_MEDIAN_UM = 3.0       # median thickness of the uniform part
UNIFORM_LOG_SD = 0.40         # spread of log(thickness); 0.4 gives ~1.5-6 um
UNIFORM_MAX_HARMONIC = 40     # how "wiggly" the thickness is along the angle
WAVINESS_AMPLITUDE_UM = 2.0   # wire out-of-roundness, metal radius varies +-2 um
PIT_DEPTH_RANGE_UM = (8.0, 16.0)    # total thickness at the bottom of a pit
PIT_WIDTH_RANGE_DEG = (8.0, 25.0)   # full width of a pit at its base

# -----------------------------------------------------------------------------
# IMAGE 6 ("06_hard_realistic"): about twice as hard as image 5 and made to
# look like the real SEM picture.  It has its OWN random generator (seed 6),
# so adding it does not change images 1-5 at all.
# -----------------------------------------------------------------------------
SEED_IMAGE_6 = 6
IMG6_NAME = "06_hard_realistic"
# Wire centre (x, y) in ImageJ pixel coordinates (x right, y DOWN).  It is
# deliberately a few px away from the image centre (300, 300).
IMG6_CENTRE_PX = (303.4, 296.8)

# --- geometry of the metal outline ---
IMG6_ELLIPSE_UM = 2.5               # slight ellipse: radius varies +-2.5 um
IMG6_ELLIPSE_ANGLE_DEG = 25.0       # direction of the long axis
IMG6_WAVINESS_UM = 4.5              # slow waviness (harmonics 2-8), peak value
IMG6_ROUGHNESS_UM = 0.5             # fine unevenness (harmonics 10-80), peak value

# --- layer thickness ---
IMG6_UNIFORM_MEDIAN_UM = 2.3        # median of the smoothly varying thin layer
IMG6_UNIFORM_LOG_SD = 0.45          # spread of log(thickness)
IMG6_UNIFORM_MAX_HARMONIC = 60      # how wiggly the thickness is along the angle
IMG6_N_THIN = 5                     # number of almost-vanishing thin stretches
IMG6_THIN_UM = (0.6, 1.0)           # thickness in those stretches (1-2 px)
IMG6_THIN_FLAT_DEG = (6.0, 14.0)    # length of the flat thin part
IMG6_THIN_TAPER_DEG = 3.0           # smooth transition at each end

# --- pits (they eat into the metal; always radially single-valued) ---
IMG6_N_PITS = 8
IMG6_DEEPEST_PIT_UM = 20.0          # one pit is exactly this deep (total thickness)
IMG6_PIT_DEPTH_RANGE_UM = (6.0, 18.0)   # total thickness at the bottom of the others
IMG6_N_NARROW_PITS = 3              # this many pits are narrow and steep ...
IMG6_NARROW_PIT_WIDTH_DEG = (4.0, 7.0)  # ... with this full width
IMG6_PIT_WIDTH_RANGE_DEG = (7.0, 16.0)  # full width of the other pits
IMG6_PIT_SHARPNESS = (0.6, 1.0)     # exponent on the pit shape; < 1 = steeper walls

# --- one broad, thick, flaky region (like the bottom of the real image) ---
IMG6_FLAKY_CENTRE_DEG = 268.0       # 270 = straight down
IMG6_FLAKY_FLAT_DEG = 28.0          # full-strength part
IMG6_FLAKY_TAPER_DEG = 10.0         # smooth transition at each side
IMG6_FLAKY_OUT_UM = 3.5             # layer bulges this far OUTWARDS ...
IMG6_FLAKY_IN_UM = 5.5              # ... and eats this far INTO the metal

# --- cracks (inside the layer: they look like resin but count as layer) ---
IMG6_N_LONG_GAPS = 3                # long dark gaps along the metal/layer interface
IMG6_LONG_GAP_DEG = (50.0, 90.0)
IMG6_N_SHORT_CRACKS = 12            # shorter interface cracks
IMG6_SHORT_CRACK_DEG = (8.0, 30.0)
IMG6_CRACK_WIDTH_PX = (1.2, 3.0)    # crack width
IMG6_N_THROUGH_CRACKS = 22          # cracks across the layer (split it into plates)
IMG6_N_FLAKY_CRACKS = 3             # extra cracks across the flaky region
IMG6_N_LAMINATION = 2               # cracks running along the flaky region (mid-layer)

# --- detached flakes outside the layer (NOT layer, label 0) ---
IMG6_N_FRAGMENTS = 4                # two of them touch each other
IMG6_FRAGMENT_GAP_UM = (1.0, 6.0)   # distance from the outer layer surface
IMG6_FRAGMENT_THICK_UM = (1.2, 3.0)
IMG6_FRAGMENT_LENGTH_DEG = (3.0, 7.0)

# --- grey values and texture ---
IMG6_GREY_RESIN = 57.0
IMG6_GREY_METAL = 143.0
IMG6_GREY_LAYER = 138.0             # base value of crack-free layer; with cracks and
                                    # darker thick parts the layer averages ~122-125
IMG6_GREY_CRACK = 15.0              # cracks are darker than resin (as in the real SEM)
IMG6_LAYER_TEXTURE_SD = 16.0        # blotchy texture of the corrosion product
IMG6_LAYER_TEXTURE_GRAIN_PX = 1.2
IMG6_PLATE_SD = 9.0                 # brightness differences between plates/flakes
IMG6_ALONG_SD = 7.0                 # slow brightness changes along the layer
IMG6_THICK_DARKEN_PER_UM = 1.6      # thick layer gets darker away from the metal ...
IMG6_THICK_DARKEN_START_UM = 3.0    # ... starting at this depth
IMG6_THIN_BRIGHTEN = 5.0            # very thin layer looks brighter (close to metal)
IMG6_EDGE_HIGHLIGHT = 55.0          # bright SEM edge effect (max, in places)
IMG6_EDGE_HIGHLIGHT_WIDTH_PX = 0.8
IMG6_METAL_GRAIN_SD = 3.0           # fine grain in the metal
IMG6_N_INCLUSIONS = 40              # small bright specks in the metal
IMG6_N_PORES = 3                    # a few small dark spots in the metal
IMG6_RESIN_TEXTURE_SD = 3.0         # slow blotches in the resin
IMG6_N_SCRATCHES = 5                # faint straight polishing scratches
IMG6_SHADING = 6.5                  # brightness slope across the image (+-)
IMG6_SHADING_BUMP = 3.0             # plus one broad brighter area
IMG6_BLUR_SIGMA_PX = 1.2
IMG6_NOISE_SD_RESIN = 6.5
IMG6_NOISE_SD_METAL = 6.0
IMG6_NOISE_SD_LAYER = 8.0
IMG6_NOISE_GRAIN_PX = 0.5           # noise is slightly correlated, like real SEM noise
IMG6_JPEG_QUALITY = 70

# The one real SEM example, only used for the side-by-side picture overview_06.png
REAL_SEM_IMAGE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "photo_2026-11-06_11-57-27.jpg"))


# =============================================================================
# SMALL HELPER FUNCTIONS FOR THE GEOMETRY
# =============================================================================

def pit_bump(theta_deg, centre_deg, width_deg):
    """A smooth bump that is 1 at the pit centre and falls to 0 at
    +-width/2 (a "cosine-squared" shape, smooth with no sharp corners).
    Angles are handled with wrap-around (359 deg is next to 0 deg)."""
    d = (theta_deg - centre_deg + 180.0) % 360.0 - 180.0   # signed angle difference
    inside = np.abs(d) < width_deg / 2.0
    return np.where(inside, np.cos(np.pi * d / width_deg) ** 2, 0.0)


def random_smooth_function(rng, max_harmonic, spectrum_falloff, min_harmonic=1):
    """Creates a random smooth periodic function f(theta) with mean 0 and
    standard deviation 1 (over the full circle).  It is a sum of sines and
    cosines; higher harmonics (faster wiggles) get smaller amplitudes.
    Only harmonics min_harmonic..max_harmonic are used (image 6 leaves out
    harmonic 1, because that would just shift the wire off its centre).
    Returns a Python function that can be evaluated at any angle(s)."""
    k = np.arange(min_harmonic, max_harmonic + 1)
    amp = 1.0 / (1.0 + (k / spectrum_falloff) ** 2)
    a = rng.normal(size=k.size) * amp
    b = rng.normal(size=k.size) * amp
    # sd of sum(a cos + b sin) over a full period is sqrt(sum(a^2+b^2)/2)
    norm = np.sqrt(np.sum(a ** 2 + b ** 2) / 2.0)

    def f(theta_deg):
        th = np.radians(np.asarray(theta_deg, dtype=float))
        out = np.zeros_like(th)
        for kk, aa, bb in zip(k, a, b):
            out += aa * np.cos(kk * th) + bb * np.sin(kk * th)
        return out / norm
    return f


def make_round_geometry(r_in_um, thickness_um):
    """Perfectly round wire with constant thickness."""
    return {
        "r_in": lambda th: np.full_like(np.asarray(th, dtype=float), r_in_um),
        "r_out": lambda th: np.full_like(np.asarray(th, dtype=float), r_in_um + thickness_um),
        "pits": [],
    }


def make_single_pit_geometry():
    """Image 3: round wire, 4 um layer, one pit 15 um deep and 20 deg wide.
    The outer surface stays on a circle; the metal surface dips inwards."""
    base_t, pit_t, centre, width = 4.0, 15.0, 45.0, 20.0
    r_out_um = METAL_RADIUS_UM + base_t

    def t(th):
        return base_t + (pit_t - base_t) * pit_bump(th, centre, width)
    return {
        "r_out": lambda th: np.full_like(np.asarray(th, dtype=float), r_out_um),
        "r_in": lambda th: r_out_um - t(th),
        "pits": [(centre, width, pit_t)],
    }


def make_realistic_geometry(rng, n_pits):
    """Images 4 and 5: slightly wavy wire + smoothly varying (lognormal-like)
    thin layer + several pits that go into the metal.

      metal surface without pits: r_surf(theta) = 140 um + waviness (+-2 um)
      uniform layer thickness:    t_u(theta)    = median * exp(log_sd * f(theta))
      outer layer surface:        r_out(theta)  = r_surf + t_u
      metal surface with pits:    r_in(theta)   = r_surf - pit depth(theta)
    so the true thickness is t = t_u + pit depth (pits eat into the metal).
    """
    # --- waviness: only harmonics 2..4 (slow, egg-like out-of-roundness)
    wav = random_smooth_function(rng, max_harmonic=4, spectrum_falloff=3.0)
    th_fine = np.arange(0, 360, 0.01)
    wav_peak = np.max(np.abs(wav(th_fine)))

    def r_surf(th):
        return METAL_RADIUS_UM + WAVINESS_AMPLITUDE_UM * wav(th) / wav_peak

    # --- smoothly varying uniform thickness with a lognormal-like distribution
    g = random_smooth_function(rng, UNIFORM_MAX_HARMONIC, spectrum_falloff=8.0)

    def t_uniform(th):
        return UNIFORM_MEDIAN_UM * np.exp(UNIFORM_LOG_SD * g(th))

    # --- pits: random positions (not overlapping), widths and depths
    pits = []
    while len(pits) < n_pits:
        centre = rng.uniform(0, 360)
        width = rng.uniform(*PIT_WIDTH_RANGE_DEG)
        depth_total = rng.uniform(*PIT_DEPTH_RANGE_UM)
        ok = all(abs((centre - c + 180) % 360 - 180) > (width + w) / 2 + 5
                 for c, w, _ in pits)
        if ok:
            pits.append((centre, width, depth_total))

    def pit_depth(th):
        # each pit is scaled so that the TOTAL thickness at its centre equals
        # the chosen depth_total (8-16 um)
        out = np.zeros_like(np.asarray(th, dtype=float))
        for centre, width, depth_total in pits:
            extra = depth_total - t_uniform(np.array([centre]))[0]
            out = out + extra * pit_bump(th, centre, width)
        return out

    return {
        "r_out": lambda th: r_surf(th) + t_uniform(th),
        "r_in": lambda th: r_surf(th) - pit_depth(th),
        "pits": pits,
    }


# =============================================================================
# ARTIFACTS FOR IMAGE 5 (cracks and detached fragments)
# Each artifact is a function that gets the polar coordinates (r, theta) of the
# sub-pixels and returns True where the artifact is.
# =============================================================================

def make_artifacts(rng, geom):
    """Returns two functions: crack_mask(r, th) and fragment_mask(r, th)."""
    r_in, r_out = geom["r_in"], geom["r_out"]

    # (a) Interface cracks: a thin resin-coloured gap just outside the metal,
    #     1-2 px wide, along a few arcs of the metal/layer boundary.
    #     The gap width tapers to zero at the ends of each arc and is never
    #     more than 40 % of the local layer thickness.
    interface_arcs = []
    for _ in range(4):
        start = rng.uniform(0, 360)
        length = rng.uniform(20, 60)
        max_w_um = rng.uniform(1.0, 2.0) * PIXEL_SIZE_UM
        interface_arcs.append((start, length, max_w_um))

    # (b) Cracks across the layer: thin (~1 px) lines running roughly
    #     radially from the metal outwards, a little tilted.
    radial_cracks = []
    for _ in range(6):
        radial_cracks.append((rng.uniform(0, 360),        # angle at the metal
                              rng.uniform(-0.15, 0.15),   # tilt (deg per um)
                              rng.uniform(0.6, 1.0) * PIXEL_SIZE_UM * 1.2,  # width
                              rng.uniform(0.6, 1.0)))     # fraction of layer crossed

    def crack_mask(r, th):
        rin, rout = r_in(th), r_out(th)
        t = rout - rin
        mask = np.zeros(r.shape, dtype=bool)
        for start, length, max_w in interface_arcs:
            d = (th - start) % 360.0
            on_arc = d < length
            taper = np.sin(np.pi * np.clip(d / length, 0, 1))  # 0 at ends, 1 in middle
            w = np.minimum(max_w * np.sqrt(taper), 0.4 * t)
            mask |= on_arc & (r >= rin) & (r < rin + w)
        for th0, tilt, width_um, frac in radial_cracks:
            depth = r - rin                      # distance above the metal
            centre_angle = th0 + tilt * depth    # tilted line
            dth = (th - centre_angle + 180.0) % 360.0 - 180.0
            arc_dist = np.abs(np.radians(dth)) * r   # distance along the arc (um)
            mask |= (arc_dist < width_um / 2) & (depth >= 0) & (depth < frac * t)
        return mask

    # (c) Detached fragments: small lens-shaped pieces of layer floating
    #     2-5 um outside the outer surface.  NOT part of the true layer.
    fragments = []
    for _ in range(2):
        fragments.append((rng.uniform(0, 360),        # angle of fragment centre
                          rng.uniform(3, 6),          # angular length (deg)
                          rng.uniform(2.0, 5.0),      # gap from layer (um)
                          rng.uniform(1.5, 3.0)))     # thickness (um)

    def fragment_mask(r, th):
        mask = np.zeros(r.shape, dtype=bool)
        for centre, length, gap, thick in fragments:
            d = (th - centre + 180.0) % 360.0 - 180.0
            shape = np.sqrt(np.clip(1 - (2 * d / length) ** 2, 0, 1))  # lens
            base = r_out(np.array([centre]))[0] + gap
            mask |= (r >= base) & (r < base + thick * shape)
        return mask

    info = {"interface_arcs": interface_arcs, "radial_cracks": radial_cracks,
            "fragments": fragments}
    return crack_mask, fragment_mask, info


# =============================================================================
# RENDERING
# =============================================================================

def polar_grid(n_px, factor, centre_px=None):
    """Polar coordinates (r in um, theta in deg) of a grid of points.
    factor = 1 -> centres of the real pixels; factor = 4 -> sub-pixel centres.
    Angle 0 = right, counter-clockwise on screen (image y axis points down).
    centre_px = (x, y) of the polar centre in ImageJ pixel coordinates
    (y points down); None means the exact image centre (images 1-5)."""
    n = n_px * factor
    coords = (np.arange(n) + 0.5) / factor          # in pixel units
    if centre_px is None:
        cx = cy = n_px / 2.0
    else:
        cx, cy = centre_px
    x = coords[np.newaxis, :] - cx
    y = cy - coords[:, np.newaxis]                  # flip: up is positive
    r = np.hypot(x, y) * PIXEL_SIZE_UM
    th = np.degrees(np.arctan2(y, x)) % 360.0
    return r, th


def classify(r, th, geom):
    """0 = resin, 1 = layer, 2 = metal at the given points."""
    rin, rout = geom["r_in"](th), geom["r_out"](th)
    cls = np.zeros(r.shape, dtype=np.uint8)
    cls[r < rout] = 1
    cls[r < rin] = 2
    return cls


def block_mean(a, f):
    """Average f x f blocks (supersampled grid -> real pixels)."""
    n = a.shape[0] // f
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


def render(geom, realistic, rng=None, artifacts=None):
    """Builds the 8-bit grey image for one geometry."""
    f = SUPERSAMPLE
    r, th = polar_grid(IMAGE_SIZE_PX, f)
    cls = classify(r, th, geom)

    # "appearance" class: what the SEM would see (cracks look like resin,
    # fragments look like layer).  The label mask ignores this.
    look = cls.copy()
    if artifacts is not None:
        crack_mask, fragment_mask = artifacts
        look[(cls == 1) & crack_mask(r, th)] = 0
        look[(cls == 0) & fragment_mask(r, th)] = 1
    del r, th

    grey = np.where(look == 2, GREY_METAL, np.where(look == 1, GREY_LAYER, GREY_RESIN))

    if realistic:
        # Blotchy texture of the corrosion product (only inside the layer)
        tex = rng.normal(size=grey.shape)
        tex = ndimage.gaussian_filter(tex, LAYER_TEXTURE_GRAIN_PX * f)
        tex *= LAYER_TEXTURE_SD / tex.std()
        grey = grey + np.where(look == 1, tex, 0.0)

    # 4x4 sub-pixels -> one pixel (anti-aliased edges)
    img = block_mean(grey, f)

    if realistic:
        img = ndimage.gaussian_filter(img, BLUR_SIGMA_PX)
        # detector noise, its strength depends on what the pixel contains
        frac = [block_mean((look == c).astype(float), f) for c in (0, 1, 2)]
        sd = frac[0] * NOISE_SD_RESIN + frac[1] * NOISE_SD_LAYER + frac[2] * NOISE_SD_METAL
        sd = ndimage.gaussian_filter(sd, BLUR_SIGMA_PX)
        img = img + rng.normal(size=img.shape) * sd

    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# =============================================================================
# GROUND-TRUTH NUMBERS
# =============================================================================

def contour_length_px(binary):
    """Perimeter of a binary region, in pixels.  The mask is slightly
    smoothed and a sub-pixel contour is traced at the 50 % level.  This avoids
    the "staircase" error of counting pixel edges (which would overestimate
    the perimeter of a circle by up to ~27 %). With 2 px smoothing the
    perimeter of the 140 um circle comes out within ~0.1 % of the exact value."""
    smooth = ndimage.gaussian_filter(binary.astype(float), 2.0)
    contours = measure.find_contours(smooth, 0.5)
    c = max(contours, key=len)                       # the main outline
    return np.sum(np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T))


def area_based_thickness_from_mask(labels):
    """Layer area / mean(inner perimeter, outer perimeter), from the label mask."""
    layer_area = np.sum(labels == 1) * PIXEL_SIZE_UM ** 2
    p_inner = contour_length_px(labels == 2) * PIXEL_SIZE_UM
    p_outer = contour_length_px(labels >= 1) * PIXEL_SIZE_UM
    return layer_area / ((p_inner + p_outer) / 2.0)


def area_based_thickness_exact(geom):
    """Same quantity, but computed exactly from the formulas (no pixels):
    area = 1/2 * integral(r_out^2 - r_in^2) dtheta,
    perimeter = integral sqrt(r^2 + (dr/dtheta)^2) dtheta."""
    th = np.arange(0, 360, 0.001)
    dth = np.radians(0.001)
    rin, rout = geom["r_in"](th), geom["r_out"](th)
    area = 0.5 * np.sum(rout ** 2 - rin ** 2) * dth

    def perim(rr):
        dr = (np.roll(rr, -1) - np.roll(rr, 1)) / (2 * dth)
        return np.sum(np.sqrt(rr ** 2 + dr ** 2)) * dth
    return area / ((perim(rin) + perim(rout)) / 2.0)


# =============================================================================
# SAVING
# =============================================================================

def save_tif(path, array):
    """8-bit TIFF with ImageJ calibration (Fiji reads 0.55 um/pixel)."""
    tifffile.imwrite(path, array, imagej=True,
                     resolution=(1.0 / PIXEL_SIZE_UM, 1.0 / PIXEL_SIZE_UM),
                     metadata={"unit": "um"})


def save_truth_csv(path, geom):
    angles = np.arange(0, 360, TRUTH_ANGLE_STEP_DEG)
    rin, rout = geom["r_in"](angles), geom["r_out"](angles)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["angle_deg", "r_inner_um", "r_outer_um", "thickness_um"])
        for a, ri, ro in zip(angles, rin, rout):
            w.writerow([f"{a:.1f}", f"{ri:.4f}", f"{ro:.4f}", f"{ro - ri:.4f}"])
    return angles, rout - rin


# =============================================================================
# IMAGE 6: "06_hard_realistic"
# A harder, more SEM-like image.  Same ground-truth definition as images 1-5,
# but the polar centre is IMG6_CENTRE_PX (not the image centre).
# =============================================================================

def flat_window(theta_deg, centre_deg, flat_deg, taper_deg):
    """1 on a flat stretch of length flat_deg around centre_deg, then falls
    smoothly (cosine-squared) to 0 over taper_deg on each side."""
    d = np.abs((np.asarray(theta_deg, dtype=float) - centre_deg + 180.0) % 360.0 - 180.0)
    x = np.clip((d - flat_deg / 2.0) / taper_deg, 0.0, 1.0)
    return np.cos(0.5 * np.pi * x) ** 2


def wrap_diff(a, b):
    """Signed angle difference a - b in degrees, in -180..180."""
    return (np.asarray(a, dtype=float) - b + 180.0) % 360.0 - 180.0


def make_geometry_6(rng):
    """Geometry of image 6.

      metal outline without pits:
          r_surf(th) = 140 um + ellipse (+-2.5) + waviness (+-4.5) + roughness (+-0.5)
      thin layer:
          t_base(th) = lognormal-like smooth thickness (median ~2.3 um), replaced
                       by 0.6-1.0 um in a few "thin stretches"
      flaky region (bottom): the layer bulges out by up to ~5 um AND eats into
          the metal by up to ~8 um, with an irregular shape
      pits: eat into the metal; total thickness at the pit bottom = chosen depth

      r_out(th) = r_surf + t_base + flaky_out
      r_in(th)  = r_surf - pit_depth - flaky_in
      thickness = r_out - r_in

    Both r_in and r_out are single-valued functions of the angle, so a straight
    line from the centre always crosses each surface exactly once (no undercut).
    """
    wav = random_smooth_function(rng, 8, 3.0, min_harmonic=2)
    rough = random_smooth_function(rng, 80, 25.0, min_harmonic=10)
    th_fine = np.arange(0, 360, 0.01)
    wav_peak = np.max(np.abs(wav(th_fine)))
    rough_peak = np.max(np.abs(rough(th_fine)))

    def r_surf(th):
        th = np.asarray(th, dtype=float)
        return (METAL_RADIUS_UM
                + IMG6_ELLIPSE_UM * np.cos(np.radians(2.0 * (th - IMG6_ELLIPSE_ANGLE_DEG)))
                + IMG6_WAVINESS_UM * wav(th) / wav_peak
                + IMG6_ROUGHNESS_UM * rough(th) / rough_peak)

    g = random_smooth_function(rng, IMG6_UNIFORM_MAX_HARMONIC, 10.0)

    def t_uniform(th):
        return IMG6_UNIFORM_MEDIAN_UM * np.exp(IMG6_UNIFORM_LOG_SD * g(th))

    # --- flaky region: irregular amounts outwards and inwards
    h_out = random_smooth_function(rng, 90, 20.0)
    h_in = random_smooth_function(rng, 90, 20.0)
    fc, ff, ft = IMG6_FLAKY_CENTRE_DEG, IMG6_FLAKY_FLAT_DEG, IMG6_FLAKY_TAPER_DEG

    def flaky_out(th):
        return IMG6_FLAKY_OUT_UM * flat_window(th, fc, ff, ft) * np.exp(0.20 * h_out(th))

    def flaky_in(th):
        return IMG6_FLAKY_IN_UM * flat_window(th, fc, ff, ft) * np.exp(0.15 * h_in(th))

    # "occupied" angular ranges, so pits, thin stretches and the flaky region
    # do not overlap: list of (centre, half-extent)
    occupied = [(fc, ff / 2 + ft)]

    def free(centre, half, margin):
        return all(abs(wrap_diff(centre, c)) > half + h + margin for c, h in occupied)

    # --- thin stretches (0.6-1.0 um), placed first
    thin = []
    while len(thin) < IMG6_N_THIN:
        c = rng.uniform(0, 360)
        flat = rng.uniform(*IMG6_THIN_FLAT_DEG)
        tt = rng.uniform(*IMG6_THIN_UM)
        half = flat / 2 + IMG6_THIN_TAPER_DEG
        if free(c, half, 6.0):
            thin.append((c, flat, tt))
            occupied.append((c, half))

    def t_base(th):
        th = np.asarray(th, dtype=float)
        tu = t_uniform(th)
        out = tu.copy()
        for c, flat, tt in thin:
            w = flat_window(th, c, flat, IMG6_THIN_TAPER_DEG)
            out = out * (1 - w) + tt * w
        return out

    # --- pits
    pits = []          # (centre, width, total depth, sharpness exponent)
    while len(pits) < IMG6_N_PITS:
        k = len(pits)
        c = rng.uniform(0, 360)
        if k < IMG6_N_NARROW_PITS:
            width = rng.uniform(*IMG6_NARROW_PIT_WIDTH_DEG)
        else:
            width = rng.uniform(*IMG6_PIT_WIDTH_RANGE_DEG)
        depth = IMG6_DEEPEST_PIT_UM if k == 0 else rng.uniform(*IMG6_PIT_DEPTH_RANGE_UM)
        q = rng.uniform(*IMG6_PIT_SHARPNESS)
        if not free(c, width / 2, 5.0):
            continue
        if depth < t_base(np.array([c]))[0] + 3.0:      # pit must be clearly deeper
            continue
        pits.append((c, width, depth, q))
        occupied.append((c, width / 2))

    def pit_depth(th):
        th = np.asarray(th, dtype=float)
        out = np.zeros_like(th)
        for c, width, depth, q in pits:
            extra = depth - t_base(np.array([c]))[0]
            out = out + extra * pit_bump(th, c, width) ** q
        return out

    return {
        "r_out": lambda th: r_surf(th) + t_base(th) + flaky_out(th),
        "r_in": lambda th: r_surf(th) - pit_depth(th) - flaky_in(th),
        "pits": [(c, w, d) for c, w, d, _ in pits],
        "pits_full": pits,
        "thin": thin,
        "flaky": (fc, ff, ft),
    }


def tabulate(fn, step_deg=0.001):
    """Fast stand-in for a geometry function: tabulate it every step_deg and
    interpolate linearly.  The error is far below 0.0001 um.  Only used for the
    rendering of the picture; the label mask and the truth table use the exact
    formulas."""
    th = np.arange(0, 360, step_deg)
    val = fn(th)
    return lambda x: np.interp(np.asarray(x, dtype=float), th, val, period=360.0)


def make_artifacts_6(rng, geom):
    """Cracks, plates, detached flakes and edge highlights for image 6.
    The functions get polar coordinates r, th (sub-pixels) plus the local
    r_in and r_out at those angles."""
    rin_t, rout_t = geom["r_in_fast"], geom["r_out_fast"]
    angles = np.arange(0, 360, 0.5)
    t_grid = rout_t(angles) - rin_t(angles)
    ps = PIXEL_SIZE_UM

    # (a) cracks/gaps along the metal/layer interface.  Their width wobbles
    #     along the arc and they never take more than 75 % of the local layer
    #     (so in very thin places the layer is mostly a dark gap, as in the
    #     real SEM image, but some layer always remains).
    arcs = []
    for n, length_rng, w_rng in ((IMG6_N_LONG_GAPS, IMG6_LONG_GAP_DEG, (2.0, 3.5)),
                                 (IMG6_N_SHORT_CRACKS, IMG6_SHORT_CRACK_DEG, IMG6_CRACK_WIDTH_PX)):
        for _ in range(n):
            arcs.append({"start": rng.uniform(0, 360), "length": rng.uniform(*length_rng),
                         "w_um": rng.uniform(*w_rng) * ps, "off_um": rng.uniform(0.0, 0.15),
                         "mod": random_smooth_function(rng, 25, 5.0)})

    # (b) cracks across the layer, preferably where the layer is thick.
    p = t_grid.copy()
    p = p / p.sum()
    through = []
    for i in range(IMG6_N_THROUGH_CRACKS):
        kind = rng.choice(["full", "from_metal", "from_outside"], p=[0.6, 0.2, 0.2])
        if kind == "full":
            f0, f1 = 0.0, 1.2
        elif kind == "from_metal":
            f0, f1 = 0.0, rng.uniform(0.5, 0.9)
        else:
            f0, f1 = rng.uniform(0.2, 0.5), 1.2
        through.append({"th0": rng.choice(angles, p=p) + rng.uniform(-0.25, 0.25),
                        "tilt": rng.uniform(-0.12, 0.12),      # deg per um of depth
                        "curv": rng.uniform(-0.01, 0.01),      # deg per um^2
                        "w_um": rng.uniform(1.0, 2.5) * ps, "f0": f0, "f1": f1})
    fc, ff, ft = geom["flaky"]
    for a in np.linspace(fc - ff / 2 - ft / 2, fc + ff / 2 + ft / 2, IMG6_N_FLAKY_CRACKS):
        through.append({"th0": a + rng.uniform(-1.5, 1.5), "tilt": rng.uniform(-0.15, 0.15),
                        "curv": rng.uniform(-0.01, 0.01),
                        "w_um": rng.uniform(1.2, 2.5) * ps, "f0": 0.0, "f1": 1.2})

    # (c) lamination cracks running along the flaky region, inside the layer
    lams = []
    for i in range(IMG6_N_LAMINATION):
        a0 = fc - ff / 2 - ft / 2 + rng.uniform(0, ff / 2)
        lams.append({"a0": a0, "length": rng.uniform(12, 22), "frac": rng.uniform(0.35, 0.65),
                     "w_um": rng.uniform(1.0, 2.0) * ps, "mod": random_smooth_function(rng, 30, 6.0)})

    def crack_mask(r, th, rin, rout):
        t = rout - rin
        depth = r - rin
        mask = np.zeros(r.shape, dtype=bool)
        for a in arcs:
            d = (th - a["start"]) % 360.0
            idx = np.nonzero((d < a["length"]) & (depth >= 0) & (depth < 3.0))[0]
            if idx.size == 0:
                continue
            taper = np.sin(np.pi * d[idx] / a["length"]) ** 0.3
            w = a["w_um"] * taper * np.clip(0.8 + 0.3 * a["mod"](th[idx]), 0.25, 1.3)
            w = np.minimum(w, 0.75 * t[idx] - a["off_um"])
            dd = depth[idx]
            mask[idx] |= (dd >= a["off_um"]) & (dd < a["off_um"] + w)
        for c in through:
            # only look at points within 10 deg of the crack (much faster)
            idx = np.nonzero((np.abs(wrap_diff(th, c["th0"])) < 10.0) & (depth >= 0))[0]
            dp = depth[idx]
            centre = c["th0"] + c["tilt"] * dp + c["curv"] * dp ** 2
            arc_dist = np.abs(np.radians(wrap_diff(th[idx], centre))) * r[idx]
            frac = dp / np.maximum(t[idx], 1e-6)
            w = c["w_um"] * (1.0 - 0.4 * np.clip(frac, 0, 1))
            mask[idx] |= (arc_dist < w / 2) & (frac >= c["f0"]) & (frac < c["f1"])
        for lm in lams:
            d = (th - lm["a0"]) % 360.0
            idx = np.nonzero(d < lm["length"])[0]
            if idx.size == 0:
                continue
            taper = np.sin(np.pi * d[idx] / lm["length"]) ** 0.5
            level = (lm["frac"] + 0.08 * lm["mod"](th[idx])) * t[idx]
            mask[idx] |= np.abs(depth[idx] - level) < lm["w_um"] * taper / 2
        return mask

    # (d) plates: each piece of layer between two cross cracks gets its own
    #     brightness offset; the outer part above a lamination crack too.
    crack_angles = np.sort(np.array([c["th0"] % 360.0 for c in through]))
    plate_offsets = rng.normal(0, IMG6_PLATE_SD, size=crack_angles.size)
    lam_offsets = rng.normal(0, IMG6_PLATE_SD, size=len(lams))

    def plate_offset(r, th, rin, rout):
        idx = np.searchsorted(crack_angles, th % 360.0) % crack_angles.size
        out = plate_offsets[idx]
        depth, t = r - rin, rout - rin
        for k, lm in enumerate(lams):
            d = (th - lm["a0"]) % 360.0
            out = out + np.where((d < lm["length"]) & (depth > lm["frac"] * t), lam_offsets[k], 0.0)
        return out

    # (e) detached flakes: 1-6 um outside the layer; two of them touch.
    frags = []
    pair_c = fc - ff / 2 - ft - rng.uniform(4, 10)
    l1, l2 = rng.uniform(*IMG6_FRAGMENT_LENGTH_DEG), rng.uniform(*IMG6_FRAGMENT_LENGTH_DEG)
    g1 = rng.uniform(1.5, 3.5)
    frags.append({"c": pair_c, "len": l1, "gap": g1, "slope": 0.0,
                  "thick": rng.uniform(*IMG6_FRAGMENT_THICK_UM)})
    frags.append({"c": pair_c + 0.92 * (l1 + l2) / 2, "len": l2, "gap": g1 + rng.uniform(0.0, 0.8),
                  "slope": rng.uniform(0.0, 0.3), "thick": rng.uniform(*IMG6_FRAGMENT_THICK_UM)})
    while len(frags) < IMG6_N_FRAGMENTS:
        c = rng.uniform(0, 360)
        if any(abs(wrap_diff(c, f["c"])) < 40 for f in frags):
            continue
        ln = rng.uniform(*IMG6_FRAGMENT_LENGTH_DEG)
        g0 = rng.uniform(IMG6_FRAGMENT_GAP_UM[0] + 0.5, IMG6_FRAGMENT_GAP_UM[1] - 0.5)
        frags.append({"c": c, "len": ln, "gap": g0, "slope": rng.uniform(-0.25, 0.25),
                      "thick": rng.uniform(*IMG6_FRAGMENT_THICK_UM)})
    for f in frags:   # keep the gap inside the allowed range along the whole flake
        lo = f["gap"] - abs(f["slope"]) * f["len"] / 2
        hi = f["gap"] + abs(f["slope"]) * f["len"] / 2
        if lo < IMG6_FRAGMENT_GAP_UM[0] or hi > IMG6_FRAGMENT_GAP_UM[1]:
            f["slope"] = 0.0
        f["c"] %= 360.0
        f["shade"] = rng.normal(0, IMG6_PLATE_SD)

    def fragment_mask(r, th, rout):
        """Returns an int array: 0 = no flake, k = inside flake number k."""
        out = np.zeros(r.shape, dtype=np.int8)
        for k, f in enumerate(frags):
            d = wrap_diff(th, f["c"])
            inside = np.abs(d) < f["len"] / 2
            shape = np.clip(1 - np.abs(2 * d / f["len"]) ** 3, 0, 1) ** (1 / 3)
            base = rout + f["gap"] + f["slope"] * d
            out[inside & (r >= base) & (r < base + f["thick"] * shape)] = k + 1
        return out

    # (f) where along the circumference the bright edge effect shows
    hl = random_smooth_function(rng, 20, 5.0)

    def highlight_strength(th):
        return IMG6_EDGE_HIGHLIGHT * np.clip(hl(th) + 0.6, 0, 1.3) / 1.3

    along = random_smooth_function(rng, 40, 8.0)

    info = {"interface_arcs": [(round(a["start"], 1), round(a["length"], 1), round(a["w_um"] / ps, 2))
                               for a in arcs],
            "through_cracks": len(through), "lamination": len(lams),
            "fragments": [(round(f["c"], 1), round(f["len"], 1), round(f["gap"], 2),
                           round(f["slope"], 2), round(f["thick"], 2)) for f in frags]}
    return {"crack_mask": crack_mask, "plate_offset": plate_offset,
            "fragment_mask": fragment_mask, "frags": frags,
            "highlight_strength": highlight_strength, "along": along, "info": info}


def render_6(geom, art, rng):
    """Builds the 8-bit picture of image 6 (see the IMG6_* parameters)."""
    f = SUPERSAMPLE
    n = IMAGE_SIZE_PX * f
    ps = PIXEL_SIZE_UM
    rin_t, rout_t = geom["r_in_fast"], geom["r_out_fast"]
    r, th = polar_grid(IMAGE_SIZE_PX, f, IMG6_CENTRE_PX)

    # only an annulus around the wire surface needs the detailed work
    tt = np.arange(0, 360, 0.01)
    r_lo = rin_t(tt).min() - 1.0
    r_hi = rout_t(tt).max() + IMG6_FRAGMENT_GAP_UM[1] + IMG6_FRAGMENT_THICK_UM[1] + 2.0
    band = (r > r_lo) & (r < r_hi)
    rb, tb = r[band], th[band]
    rin_b, rout_b = rin_t(tb), rout_t(tb)

    # classes: 0 resin, 1 layer, 2 metal, 3 crack (looks dark), 4 detached flake
    cls_b = np.zeros(rb.shape, dtype=np.uint8)
    cls_b[rb < rout_b] = 1
    cls_b[rb < rin_b] = 2
    look_b = cls_b.copy()
    look_b[(cls_b == 1) & art["crack_mask"](rb, tb, rin_b, rout_b)] = 3
    frag_id = art["fragment_mask"](rb, tb, rout_b)
    look_b[(cls_b == 0) & (frag_id > 0)] = 4
    look = np.where(r <= r_lo, 2, 0).astype(np.uint8)
    look[band] = look_b

    def up(a):      # pixel-resolution field -> sub-pixel grid
        return np.repeat(np.repeat(a, f, axis=0), f, axis=1)

    def smooth_noise(shape, sigma, sd):
        x = ndimage.gaussian_filter(rng.normal(size=shape), sigma)
        return x * (sd / x.std())

    grey = np.zeros((n, n))

    # resin: slow blotches
    resin_tex = up(smooth_noise((IMAGE_SIZE_PX,) * 2, 20.0, IMG6_RESIN_TEXTURE_SD)
                   + smooth_noise((IMAGE_SIZE_PX,) * 2, 2.0, 1.5))
    grey[look == 0] = IMG6_GREY_RESIN + resin_tex[look == 0]
    del resin_tex

    # metal: fine grain
    metal_tex = up(smooth_noise((IMAGE_SIZE_PX,) * 2, 0.7, IMG6_METAL_GRAIN_SD))
    grey[look == 2] = IMG6_GREY_METAL + metal_tex[look == 2]
    del metal_tex

    # cracks
    grey[look == 3] = IMG6_GREY_CRACK

    # layer and flakes: base + plate offsets + slow changes along the layer
    # + darker deep inside thick layer + brighter where very thin + texture
    tex = smooth_noise((n, n), IMG6_LAYER_TEXTURE_GRAIN_PX * f, IMG6_LAYER_TEXTURE_SD)
    lay = look_b == 1
    depth = rb[lay] - rin_b[lay]
    t_loc = rout_b[lay] - rin_b[lay]
    g_layer = (IMG6_GREY_LAYER
               + art["plate_offset"](rb[lay], tb[lay], rin_b[lay], rout_b[lay])
               + IMG6_ALONG_SD * art["along"](tb[lay])
               - IMG6_THICK_DARKEN_PER_UM * np.maximum(depth - IMG6_THICK_DARKEN_START_UM, 0)
               + IMG6_THIN_BRIGHTEN * np.clip((2.0 - t_loc) / 1.4, 0, 1))
    fr = look_b == 4
    shades = np.array([0.0] + [fg["shade"] for fg in art["frags"]])
    g_frag = IMG6_GREY_LAYER + shades[frag_id[fr]] + IMG6_ALONG_SD * art["along"](tb[fr])
    gb = grey[band]
    gb[lay] = g_layer
    gb[fr] = g_frag
    gb[lay | fr] += tex[band][lay | fr]
    grey[band] = gb
    del tex

    # bright SEM edge effect on layer/flake pixels next to resin or cracks
    solid = (look == 1) | (look == 4)
    dist = ndimage.distance_transform_edt(solid) / f          # in pixels
    hl = np.zeros((n, n))
    hl[band] = art["highlight_strength"](tb)
    grey += np.where(solid, hl * np.exp(-(dist - 0.125) / IMG6_EDGE_HIGHLIGHT_WIDTH_PX), 0.0)
    del dist, hl, solid

    # bright inclusions and a few dark pores in the metal (kept 3 um away
    # from the metal surface)
    yy, xx = np.mgrid[0:n, 0:n]
    k_incl = 0
    while k_incl < IMG6_N_INCLUSIONS + IMG6_N_PORES:
        rr = 125.0 * np.sqrt(rng.uniform())
        aa = rng.uniform(0, 360)
        if rr > rin_t(np.array([aa]))[0] - 3.0:
            continue
        cx = (IMG6_CENTRE_PX[0] + rr / ps * np.cos(np.radians(aa))) * f
        cy = (IMG6_CENTRE_PX[1] - rr / ps * np.sin(np.radians(aa))) * f
        if k_incl < IMG6_N_INCLUSIONS:
            rad = rng.uniform(0.3, 1.2) * f
            amp = rng.uniform(40, 100)
        else:
            rad = rng.uniform(1.0, 2.0) * f
            amp = -rng.uniform(40, 70)
        x0, x1 = int(cx - rad - 2), int(cx + rad + 3)
        y0, y1 = int(cy - rad - 2), int(cy + rad + 3)
        d = np.hypot(xx[y0:y1, x0:x1] + 0.5 - cx, yy[y0:y1, x0:x1] + 0.5 - cy)
        grey[y0:y1, x0:x1] += amp * np.clip(rad + 0.5 - d, 0, 1)
        k_incl += 1
    del yy, xx

    # sub-pixels -> pixels (anti-aliased)
    img = block_mean(grey, f)
    frac = {c: block_mean((look == c).astype(float), f) for c in range(5)}
    del grey, r, th

    # faint straight polishing scratches (strong in resin, weak on the metal)
    yp, xp = np.mgrid[0:IMAGE_SIZE_PX, 0:IMAGE_SIZE_PX] + 0.5
    for _ in range(IMG6_N_SCRATCHES):
        phi = np.radians(rng.uniform(0, 180))
        px, py = rng.uniform(0, IMAGE_SIZE_PX, size=2)
        width = rng.uniform(0.5, 1.0)
        amp = rng.choice([-1, 1]) * rng.uniform(4, 8)
        dist = np.abs((xp - px) * np.sin(phi) - (yp - py) * np.cos(phi))
        along = (xp - px) * np.cos(phi) + (yp - py) * np.sin(phi)
        fade = np.clip(1 - np.abs(along) / rng.uniform(150, 400), 0, 1)
        img += (amp * np.exp(-0.5 * (dist / width) ** 2) * fade
                * (frac[0] + 0.3 * frac[2]))

    # uneven illumination: a slope across the image plus one broad bright area
    sphi = np.radians(rng.uniform(0, 360))
    u = ((xp - IMAGE_SIZE_PX / 2) * np.cos(sphi) + (yp - IMAGE_SIZE_PX / 2) * np.sin(sphi)) \
        / (IMAGE_SIZE_PX / 2)
    bx, by = rng.uniform(100, 500, size=2)
    img += IMG6_SHADING * np.clip(u, -1, 1) \
        + IMG6_SHADING_BUMP * np.exp(-((xp - bx) ** 2 + (yp - by) ** 2) / (2 * 120.0 ** 2))

    # blur, then slightly correlated detector noise
    img = ndimage.gaussian_filter(img, IMG6_BLUR_SIGMA_PX)
    sd = ((frac[0] + frac[3]) * IMG6_NOISE_SD_RESIN + (frac[1] + frac[4]) * IMG6_NOISE_SD_LAYER
          + frac[2] * IMG6_NOISE_SD_METAL)
    sd = ndimage.gaussian_filter(sd, IMG6_BLUR_SIGMA_PX)
    img = img + smooth_noise(img.shape, IMG6_NOISE_GRAIN_PX, 1.0) * sd
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def labels_6(geom):
    """Ideal label mask of image 6 (0 resin, 1 layer, 2 metal), decided at the
    pixel centres with the EXACT formulas, around IMG6_CENTRE_PX."""
    r, th = polar_grid(IMAGE_SIZE_PX, 1, IMG6_CENTRE_PX)
    tt = np.arange(0, 360, 0.01)
    r_lo = geom["r_in_fast"](tt).min() - 2.0
    r_hi = geom["r_out_fast"](tt).max() + 2.0
    labels = np.where(r <= r_lo, 2, 0).astype(np.uint8)
    band = (r > r_lo) & (r < r_hi)
    labels[band] = classify(r[band], th[band], geom)
    return labels


def make_image_6():
    """Builds everything for image 6.  Uses its own random generator."""
    rng6 = np.random.default_rng(SEED_IMAGE_6)
    geom = make_geometry_6(rng6)
    geom["r_in_fast"] = tabulate(geom["r_in"])
    geom["r_out_fast"] = tabulate(geom["r_out"])
    art = make_artifacts_6(rng6, geom)
    img = render_6(geom, art, rng6)
    labels = labels_6(geom)
    return geom, art, img, labels


def make_overview_6(item5, item6, path):
    """overview_06.png: real SEM (left half), 05 and 06 at the same scale,
    zoomed crops of a thin and a cracked region for each, plus the labels and
    thickness vs angle of 06."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = PIXEL_SIZE_UM
    real = None
    if os.path.exists(REAL_SEM_IMAGE):
        real = np.asarray(Image.open(REAL_SEM_IMAGE).convert("L"))[:595, :612]

    def point(centre, r_um, a_deg):
        return (centre[0] + r_um / ps * np.cos(np.radians(a_deg)),
                centre[1] - r_um / ps * np.sin(np.radians(a_deg)))

    # crop centres (x, y) in px: thin region and cracked region
    c5 = (IMAGE_SIZE_PX / 2, IMAGE_SIZE_PX / 2)
    c6 = IMG6_CENTRE_PX
    crops = {
        "real": [(43, 300), (300, 532)],                # left edge (thin), bottom (cracked)
        "05": [point(c5, item5["r_out"](item5["thin_angle"]), item5["thin_angle"]),
               point(c5, item5["r_out"](item5["crack_angle"]), item5["crack_angle"])],
        "06": [point(c6, item6["r_out"](item6["thin_angle"]), item6["thin_angle"]),
               point(c6, item6["r_out"](item6["crack_angle"]), item6["crack_angle"])],
    }
    half = 55   # crop half-size in px (60 um crop)

    fig = plt.figure(figsize=(15, 21))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.62, 0.62, 0.75])
    cols = [("real SEM (left half of photo)", real, "real"),
            ("05_realistic_with_artifacts", item5["img"], "05"),
            ("06_hard_realistic", item6["img"], "06")]
    for j, (title, im, key) in enumerate(cols):
        ax = fig.add_subplot(gs[0, j])
        if im is None:
            ax.text(0.5, 0.5, "real image not found", ha="center"); ax.axis("off"); continue
        hgt, wid = im.shape
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, extent=(0, wid * ps, hgt * ps, 0),
                  interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("um", fontsize=8)
        ax.tick_params(labelsize=7)
        for k, (cx, cy) in enumerate(crops[key]):
            ax.add_patch(plt.Rectangle(((cx - half) * ps, (cy - half) * ps), 2 * half * ps,
                                       2 * half * ps, fill=False, ec=["#3fa7ff", "#ff6a3f"][k], lw=1.2))
        for k, label in enumerate(["thin region", "cracked region"]):
            ax2 = fig.add_subplot(gs[1 + k, j])
            cx, cy = crops[key][k]
            x0, y0 = int(round(cx - half)), int(round(cy - half))
            x0, y0 = max(0, min(x0, wid - 2 * half)), max(0, min(y0, hgt - 2 * half))
            ax2.imshow(im[y0:y0 + 2 * half, x0:x0 + 2 * half], cmap="gray", vmin=0, vmax=255,
                       extent=(0, 2 * half * ps, 2 * half * ps, 0), interpolation="nearest")
            ax2.set_title(f"{key}: {label} (60 x 60 um)", fontsize=10,
                          color=["#1f77d4", "#d4471f"][k])
            ax2.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(item6["labels"], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
    ax.plot([c6[0] - 0.5], [c6[1] - 0.5], "r+", ms=10)
    ax.set_title("06 labels (0 resin, 1 layer, 2 metal); + = centre", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[3, 1:])
    ax.plot(item6["angles"], item6["t"], color="#1f5fa8", lw=1.0, label="06")
    ax.plot(item5["angles"], item5["t"], color="#999999", lw=0.8, alpha=0.8, label="05")
    ax.axhline(1.1, color="#d4471f", lw=0.8, ls="--", label="1.1 um (2 px)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max(22, item6["t"].max() * 1.08))
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("angle (deg, 0 = right, CCW)")
    ax.set_ylabel("true radial thickness (um)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # optional: python generate_synthetic.py <output folder>
    global OUTPUT_DIR
    if len(sys.argv) > 1:
        OUTPUT_DIR = os.path.abspath(sys.argv[1])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    r_px, th_px = polar_grid(IMAGE_SIZE_PX, 1)   # pixel centres, for labels

    # ---- define the five test cases ----------------------------------------
    rng4 = np.random.default_rng(SEED_IMAGE_4)
    rng5 = np.random.default_rng(SEED_IMAGE_5)
    geom4 = make_realistic_geometry(rng4, n_pits=5)
    n_pits_5 = int(rng5.integers(4, 7))          # 4, 5 or 6 pits
    geom5 = make_realistic_geometry(rng5, n_pits=n_pits_5)
    crack5, frag5, art_info = make_artifacts(rng5, geom5)

    cases = [
        ("01_uniform_5um", make_round_geometry(METAL_RADIUS_UM, 5.0), False, None, None),
        ("02_uniform_10um_noisy", make_round_geometry(METAL_RADIUS_UM, 10.0), True,
         np.random.default_rng(SEED_IMAGE_2), None),
        ("03_single_pit", make_single_pit_geometry(), True,
         np.random.default_rng(SEED_IMAGE_3), None),
        ("04_mixed_uniform_and_pits", geom4, True, rng4, None),
        ("05_realistic_with_artifacts", geom5, True, rng5, (crack5, frag5)),
    ]

    summary = []
    overview = []
    for name, geom, realistic, rng, artifacts in cases:
        print(f"Making {name} ...")
        img = render(geom, realistic, rng, artifacts)
        labels = classify(r_px, th_px, geom)

        save_tif(os.path.join(OUTPUT_DIR, f"{name}.tif"), img)
        save_tif(os.path.join(OUTPUT_DIR, f"{name}_labels.tif"), labels)
        if artifacts is not None:
            Image.fromarray(img).save(os.path.join(OUTPUT_DIR, f"{name}.jpg"),
                                      quality=JPEG_QUALITY)
        angles, t = save_truth_csv(os.path.join(OUTPUT_DIR, f"{name}_truth.csv"), geom)

        summary.append({
            "name": name,
            "pixel_size_um": PIXEL_SIZE_UM,
            "mean_thickness_um": t.mean(),
            "median_thickness_um": np.median(t),
            "min_thickness_um": t.min(),
            "max_thickness_um": t.max(),
            "sd_thickness_um": t.std(),
            "area_based_mean_thickness_um": area_based_thickness_from_mask(labels),
            "area_based_mean_thickness_exact_um": area_based_thickness_exact(geom),
            "n_pits": len(geom["pits"]),
        })
        overview.append((name, img, labels, angles, t))

    # ---- image 6 (own random generator, so images 1-5 are unaffected) ------
    print(f"Making {IMG6_NAME} ...")
    geom6, art6, img6, labels6 = make_image_6()
    save_tif(os.path.join(OUTPUT_DIR, f"{IMG6_NAME}.tif"), img6)
    save_tif(os.path.join(OUTPUT_DIR, f"{IMG6_NAME}_labels.tif"), labels6)
    Image.fromarray(img6).save(os.path.join(OUTPUT_DIR, f"{IMG6_NAME}.jpg"),
                               quality=IMG6_JPEG_QUALITY)
    angles6, t6 = save_truth_csv(os.path.join(OUTPUT_DIR, f"{IMG6_NAME}_truth.csv"), geom6)
    summary.append({
        "name": IMG6_NAME,
        "pixel_size_um": PIXEL_SIZE_UM,
        "mean_thickness_um": t6.mean(),
        "median_thickness_um": np.median(t6),
        "min_thickness_um": t6.min(),
        "max_thickness_um": t6.max(),
        "sd_thickness_um": t6.std(),
        "area_based_mean_thickness_um": area_based_thickness_from_mask(labels6),
        "area_based_mean_thickness_exact_um": area_based_thickness_exact(geom6),
        "n_pits": len(geom6["pits"]),
    })

    # overview_06.png: compare with the real SEM image and with image 5
    name5, img5, labels5, angles5, t5 = overview[4]
    arc5 = art_info["interface_arcs"][0]
    fc6 = geom6["flaky"][0]
    item5 = {"img": img5, "angles": angles5, "t": t5,
             "r_out": lambda a: geom5["r_out"](np.array([a]))[0],
             "thin_angle": float(angles5[np.argmin(t5)]),
             "crack_angle": float((arc5[0] + arc5[1] / 2) % 360)}
    item6 = {"img": img6, "labels": labels6, "angles": angles6, "t": t6,
             "r_out": lambda a: geom6["r_out"](np.array([a]))[0],
             "thin_angle": float(geom6["thin"][int(np.argmin([x[2] for x in geom6["thin"]]))][0]),
             "crack_angle": float(fc6 - 8.0)}
    make_overview_6(item5, item6, os.path.join(OUTPUT_DIR, "overview_06.png"))

    with open(os.path.join(OUTPUT_DIR, "ground_truth_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    make_overview(overview, os.path.join(OUTPUT_DIR, "overview.png"))

    print("\nPits:")
    print("  04:", [(round(c, 1), round(w, 1), round(d, 2)) for c, w, d in geom4["pits"]])
    print("  05:", [(round(c, 1), round(w, 1), round(d, 2)) for c, w, d in geom5["pits"]])
    print("  05 artifacts:", art_info)
    print("  06 pits (centre, width, depth, sharpness):",
          [tuple(round(float(v), 2) for v in p) for p in geom6["pits_full"]])
    print("  06 thin stretches (centre, length, thickness):",
          [tuple(round(float(v), 2) for v in p) for p in geom6["thin"]])
    print("  06 flaky region (centre, flat, taper):", geom6["flaky"])
    print("  06 artifacts:", art6["info"])
    print(f"  06 centre: x = {IMG6_CENTRE_PX[0]} px, y = {IMG6_CENTRE_PX[1]} px (ImageJ)")
    print("\nSummary:")
    for row in summary:
        print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


def make_overview(items, path):
    """Contact sheet: images (top), label masks (middle), thickness vs angle (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(items), figsize=(4 * len(items), 11.5),
                             gridspec_kw={"height_ratios": [1, 1, 0.75]})
    for col, (name, img, labels, angles, t) in enumerate(items):
        ax = axes[0, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        ax = axes[1, col]
        ax.imshow(labels, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        ax.set_title("labels (0 resin, 1 layer, 2 metal)", fontsize=8)
        ax.axis("off")
        ax = axes[2, col]
        ax.plot(angles, t, color="#1f5fa8", lw=1.2)
        ax.set_xlim(0, 360)
        ax.set_ylim(0, max(18, t.max() * 1.1))
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("angle (deg, 0 = right, CCW)", fontsize=8)
        if col == 0:
            ax.set_ylabel("true thickness (um)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
