"""Recover clean binary segmentation masks from the ImageJ-annotated
"masques" images.

The files in data/masques/*.jpg are NOT binary masks. They are the original
plate photo with a thin, hand-traced, reddish-colored outline drawn on top
to mark the region of interest (e.g. the wound/gap area in a scratch assay).
Some traces are fully closed loops; others exit the image and rely on the
image border to close the shape (a common ImageJ selection pattern).

This script:
  1. Detects the annotation line via color difference (image vs masque)
     filtered by hue/saturation (the line is a saturated red, unlike the
     mostly-desaturated microscopy background), which discards JPEG
     recompression noise that a plain pixel-diff threshold picks up.
  2. Drops small isolated noise blobs.
  3. Closes small gaps in the line (JPEG compression can fade a 1-2px line).
  4. Closes the shape across the image border wherever the line exits the
     frame (connects the two border-crossing points along that edge).
  5. Fills the enclosed interior via flood-fill-from-outside.
  6. Keeps only the largest connected foreground region.

Outputs:
  - data/masks_binary/{id}.png   -- binary mask (0/255), one per image
  - outputs/qc_masks/{id}.jpg    -- QC overlay (red = mask) for manual review
  - outputs/qc_masks/_area_fractions.csv -- foreground area fraction per image,
    sorted ascending, to help spot failures quickly
"""
import csv
import glob
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

IMAGES_DIR = "data/images"
MASQUES_DIR = "data/masques"
OUT_MASKS_DIR = "data/masks_binary"
QC_DIR = "outputs/qc_masks"

DIFF_THRESHOLD = 20
SAT_THRESHOLD = 70
HUE_MAX = 20  # keep hues in [0, HUE_MAX] or [179-HUE_MAX, 179] (red wraps around 0)
MIN_BLOB_AREA = 15
CLOSE_KERNEL = 9
CLOSE_ITERS = 2
MAX_BRIDGE_GAP = 90  # max px gap to bridge between broken line fragments (JPEG fading)


def bridge_line_fragments(line, max_gap=MAX_BRIDGE_GAP):
    """JPEG compression can fade the thin annotation line below our
    detection threshold in places, breaking one closed loop into several
    disconnected fragments. Repeatedly connect the two closest remaining
    fragments with a straight segment until they form a single component
    (or the closest remaining gap exceeds max_gap, in which case we stop
    and let the caller flag it for manual QC)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    if n <= 2:
        return line, True

    comp_ids = list(range(1, n))
    pts = {i: np.column_stack(np.where(labels == i)[::-1]) for i in comp_ids}  # (x, y)
    merged = line.copy()
    parent = {i: i for i in comp_ids}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    while True:
        groups = {}
        for i in comp_ids:
            groups.setdefault(find(i), []).append(i)
        if len(groups) <= 1:
            return merged, True

        group_ids = list(groups.keys())
        best = None
        for gi in range(len(group_ids)):
            for gj in range(gi + 1, len(group_ids)):
                pa = np.vstack([pts[c] for c in groups[group_ids[gi]]])
                pb = np.vstack([pts[c] for c in groups[group_ids[gj]]])
                d, idx = cKDTree(pb).query(pa, k=1)
                m = int(d.argmin())
                if best is None or d[m] < best[0]:
                    best = (d[m], tuple(pa[m]), tuple(pb[idx[m]]), group_ids[gi], group_ids[gj])

        dist, pt_a, pt_b, ga, gb = best
        if dist > max_gap:
            return merged, False
        cv2.line(merged, pt_a, pt_b, 255, thickness=2)
        parent[find(ga)] = find(gb)


def extract_mask(image_bgr, masque_bgr):
    h = min(image_bgr.shape[0], masque_bgr.shape[0])
    w = min(image_bgr.shape[1], masque_bgr.shape[1])
    image_bgr = image_bgr[:h, :w]
    masque_bgr = masque_bgr[:h, :w]

    diff = np.abs(image_bgr.astype(np.int16) - masque_bgr.astype(np.int16)).max(axis=2).astype(np.uint8)
    hsv = cv2.cvtColor(masque_bgr, cv2.COLOR_BGR2HSV)
    sat, hue = hsv[:, :, 1], hsv[:, :, 0]
    hue_ok = (hue < HUE_MAX) | (hue > 179 - HUE_MAX)
    line = ((diff > DIFF_THRESHOLD) & (sat > SAT_THRESHOLD) & hue_ok).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    clean = np.zeros_like(line)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
            clean[labels == i] = 255
    line = clean

    line, fully_bridged = bridge_line_fragments(line)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    wall = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

    def close_edge(edge):
        pts = np.where(edge > 0)[0]
        if len(pts) >= 2:
            edge[pts.min():pts.max() + 1] = 255

    close_edge(wall[0, :])
    close_edge(wall[-1, :])
    close_edge(wall[:, 0])
    close_edge(wall[:, -1])

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    outside = wall.copy()
    cv2.floodFill(outside, flood_mask, (0, 0), 255)
    interior = cv2.bitwise_not(outside)
    fg = cv2.bitwise_or(interior, wall)

    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n2 > 1:
        largest = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels2 == largest, 255, 0).astype(np.uint8)

    # Sanity check: if the flood-fill leaked out through an undetected micro-gap
    # in the wall, `interior` stays near-empty and `fg` collapses to ~just the
    # traced line itself (fg_area comparable to wall_area instead of much
    # larger). Flag this the same way as an unbridgeable gap so it surfaces in
    # the QC warnings instead of silently producing an unfilled mask.
    wall_area = int((wall > 0).sum())
    fg_area = int((fg > 0).sum())
    fill_leaked = fg_area < 3 * wall_area

    return image_bgr, fg, fully_bridged and not fill_leaked


def save_qc_overlay(image_bgr, mask, out_path):
    overlay = image_bgr.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(image_bgr, 0.55, overlay, 0.45, 0)
    cv2.imwrite(out_path, blended)


def main():
    os.makedirs(OUT_MASKS_DIR, exist_ok=True)
    os.makedirs(QC_DIR, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(IMAGES_DIR, "*")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    areas = []
    unresolved = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        masque_path = os.path.join(MASQUES_DIR, f"{stem}.jpg")
        if not os.path.exists(masque_path):
            print(f"[WARN] no masque for {stem}, skipping")
            continue

        image_bgr = cv2.imread(img_path)
        masque_bgr = cv2.imread(masque_path)
        image_bgr, mask, fully_bridged = extract_mask(image_bgr, masque_bgr)

        cv2.imwrite(os.path.join(OUT_MASKS_DIR, f"{stem}.png"), mask)
        save_qc_overlay(image_bgr, mask, os.path.join(QC_DIR, f"{stem}.jpg"))

        areas.append((stem, (mask > 0).mean()))
        if not fully_bridged:
            unresolved.append(stem)

    if unresolved:
        print(f"[WARN] {len(unresolved)} image(s) had a gap too large to bridge automatically: {unresolved}")
        print("        -> inspect their QC overlays, the traced outline is likely incomplete.")

    areas.sort(key=lambda x: x[1])
    with open(os.path.join(QC_DIR, "_area_fractions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "area_fraction"])
        writer.writerows(areas)

    print(f"Processed {len(areas)} images.")
    print("5 smallest foreground fractions (check these first):")
    for stem, frac in areas[:5]:
        print(f"  {stem}: {frac:.4f}")
    print("5 largest foreground fractions (check these first):")
    for stem, frac in areas[-5:]:
        print(f"  {stem}: {frac:.4f}")


if __name__ == "__main__":
    main()
