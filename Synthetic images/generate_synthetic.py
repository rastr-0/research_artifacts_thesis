#!/usr/bin/env python3
"""
generate_synthetic.py
=====================

Makes 5 synthetic "SEM cross-section" images of a thin magnesium wire with a
corrosion layer around it.  Because we build the images ourselves from simple
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

Run it with:   python generate_synthetic.py
(needs numpy, scipy, scikit-image, pillow, matplotlib, tifffile)
"""

import csv
import os

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


def random_smooth_function(rng, max_harmonic, spectrum_falloff):
    """Creates a random smooth periodic function f(theta) with mean 0 and
    standard deviation 1 (over the full circle).  It is a sum of sines and
    cosines; higher harmonics (faster wiggles) get smaller amplitudes.
    Returns a Python function that can be evaluated at any angle(s)."""
    k = np.arange(1, max_harmonic + 1)
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

def polar_grid(n_px, factor):
    """Polar coordinates (r in um, theta in deg) of a grid of points.
    factor = 1 -> centres of the real pixels; factor = 4 -> sub-pixel centres.
    Angle 0 = right, counter-clockwise on screen (image y axis points down)."""
    n = n_px * factor
    coords = (np.arange(n) + 0.5) / factor          # in pixel units
    centre = n_px / 2.0
    x = coords[np.newaxis, :] - centre
    y = centre - coords[:, np.newaxis]              # flip: up is positive
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
# MAIN
# =============================================================================

def main():
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
