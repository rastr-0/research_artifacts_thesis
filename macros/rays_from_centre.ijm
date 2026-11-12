// Rays from the centre: thickness of the corrosion layer every 0.5 degrees.
//
// Input: an open LABELS image (0 = resin, 1 = layer, 2 = metal), e.g.
//        synthetic/05_realistic_with_artifacts_labels.tif. Calibration is
//        taken from Image > Properties (should say 0.55 micron).
// How:   from the centre of the wire, walk outwards along a straight line
//        and count how long the line stays inside the layer (label 1).
//        Repeat every 0.5 degrees.
// Output: a table "Rays" (angle_deg, thickness) that you can save as CSV,
//         plus a summary (mean, median, min, max, SD) in the Log window.
// Angles: 0 deg = pointing right, increasing counter-clockwise
//         (same convention as the *_truth.csv files).

angleStep = 0.1;   // degrees between rays
stepAlong = 0.05;   // step along each ray, in pixels (smaller = more precise, slower)

getPixelSize(unit, pw, ph);
w = getWidth();
h = getHeight();

// Centre of the wire = centroid of metal + layer (labels 1 and 2)
setThreshold(1, 2);
run("Create Selection");
List.setMeasurements;
cx = List.getValue("X");
cy = List.getValue("Y");
toUnscaled(cx, cy);          // convert from microns to pixel coordinates
run("Select None");
resetThreshold();

n = 360 / angleStep;
angles = newArray(n);
thick = newArray(n);
rMax = sqrt(w*w + h*h) / 2;

setBatchMode(true);
for (i = 0; i < n; i++) {
    a = i * angleStep * PI / 180;
    inside = 0;
    r = 0;
    while (r < rMax) {
        // ImageJ centroid coordinates: pixel (i, j) covers i..i+1, so floor() picks the pixel
        x = floor(cx + r * cos(a));
        y = floor(cy - r * sin(a));          // minus: image y axis points down
        if (x < 0 || y < 0 || x >= w || y >= h) {
            r = rMax;                        // left the image: stop this ray
        } else {
            if (getPixel(x, y) == 1) inside++;
            r += stepAlong;
        }
    }
    angles[i] = i * angleStep;
    thick[i] = inside * stepAlong * pw;      // length inside the layer, in calibrated units
}
setBatchMode(false);

// Table with one row per angle
Table.create("Rays");
Table.setColumn("angle_deg", angles);
Table.setColumn("thickness_" + unit, thick);
Table.update;

// Summary
Array.getStatistics(thick, mn, mx, mean, sd);
sorted = Array.copy(thick);
Array.sort(sorted);
median = (sorted[n/2 - 1] + sorted[n/2]) / 2;
print("Rays from centre: " + getTitle());
print("  centre (px): " + d2s(cx, 1) + ", " + d2s(cy, 1));
print("  mean " + d2s(mean, 2) + "  median " + d2s(median, 2) + "  min " + d2s(mn, 2) +
      "  max " + d2s(mx, 2) + "  SD " + d2s(sd, 2) + "  (" + unit + ")");
