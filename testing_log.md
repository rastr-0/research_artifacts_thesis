# Testing Log: Thickness Measurement Methods

Existing ImageJ/Fiji thickness tools compared on synthetic images with a known thickness.

## Setup
- **Images:** `Synthetic images/` (generator: `generate_synthetic.py`). A metal wire (~290 µm diameter) with a thin corrosion ring; pixel size 0.55 µm.
- **Input:** the ideal masks (`*_labels.tif`: 0 resin, 1 layer, 2 metal). These tests cover the **measuring step only**; segmentation is not tested yet.
- **Ground truth:** `*_truth.csv`, thickness measured radially from the wire centre every 0.5°.
- **Main test image:** `05_realistic_with_artifacts`, with true thickness mean 4.48 µm, median 3.51 µm and range 1.21–15.92 µm (6 pits).

## Results (image 05)

| # | Method | Mean (µm) | Median (µm) | Max (µm) | Error of mean |
|---|---|---|---|---|---|
| – | **Ground truth** | **4.48** | **3.51** | **15.92** | – |
| 1 | Local Thickness (Fiji), whole ring | 6.79 | 5.08 | 15.28 | +52% |
| 2 | Area method (area ÷ perimeter / equivalent radii) | 4.15 / 4.45 | – | – | −5% / +2%* |
| 3 | Local Thickness, read along the metal edge | 5.21 | 3.97 | 15.28 | +16% |
| 4 | Rays from the centre (own macro), 0.5° | 4.46 | 3.47 | 15.95 | −0.4% |
| 5 | ThicknessTool (Maidana et al. 2020) | – | – | – | failed |
| 6 | asc-LayerThickness "Rings", outwards | 4.37 | 3.35 | 15.86 | −2% |
| 6 | asc-LayerThickness "Rings", inwards | 4.05 | 3.30 | 13.85 | −10% |
| 7 | BoneJ Slice Geometry, 2D thickness | 7.21 | – | 15.28 | +61% |

\* Compared with the true area-based mean (4.37 µm).

## Notes per method
1. **Local Thickness:** gives one value per pixel, so thick pits (many pixels) are over-counted. It also adds about +1 px.
   - On image 01 (constant 5.00 µm) it gives a mean of 5.46 µm.
   - Values come only in discrete steps of 2·√n px.
2. **Area method:** a good average, but only one number per image. Fiji perimeters of pixelated curves are about 5% too long.
3. **Local Thickness along the edge:** sampling along the metal edge removes most of the per-pixel bias. The remaining error is mostly the +1 px offset.
4. **Rays from the centre:** per angle, the mean error is −0.01 µm, the mean absolute error 0.21 µm, and 98% of angles are within 1 px. Finer angle steps change nothing, so the limit is the pixelated mask. Caveat: the ground truth uses the same radial definition.
5. **ThicknessTool:** a hidden median filter (radius 10 px) erases layers thinner than about 10 px, so only one thick pit was measured. A patched copy ran out of memory. It is not suitable for thin layers.
6. **asc-LayerThickness:** the closest-point distance depends on direction.
   - Outwards is slightly low (−0.21 µm per point).
   - Inwards underestimates pits by up to 4 µm, because the nearest metal is the pit's rim, not its bottom.
7. **BoneJ Slice Geometry:** the same Local Thickness algorithm, averaged over all pixels, so it has the same problems as method 1.

## Conclusions
- **Best measuring approach:** rays from the centre (unbiased, within about 1 px), followed by asc-LayerThickness outwards.
- **Not suitable for thin layers:** Local Thickness and BoneJ (per-pixel bias and +1 px offset), and ThicknessTool.
- **Limit of the measuring step:** about 1 px with a pixel mask. A sub-pixel study suggests refining the edges on the grey image can improve this, especially at the layer/resin edge.
- **Next:** segmentation. With a realistic (non-ideal) mask the error per angle grows to about 0.6–0.9 µm, so segmentation, not measurement, will dominate the total error.
