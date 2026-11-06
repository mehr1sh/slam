# Standalone ORB Extractor + Dataset Benchmark

This is the **actual scratch ORB feature-extraction code extracted from the provided `slam-main.zip`**, packaged independently of the SLAM / PnP / ICP pipeline. The C++ files under `include/features/` and `src/features/` are byte-for-byte copies from that archive. The small CLI adapter and Python benchmark scripts are new.

**What the original implementation does:** FAST-9 keypoint detection + non-maximum suppression, intensity-centroid orientation, a fixed-seed *project-specific* 256-bit rotated BRIEF pattern, optional scale pyramid, and (in the included library) brute-force Hamming matching. Unlike OpenCV ORB, the scratch implementation does not use OpenCV's learned BRIEF pattern, Harris keypoint re-ranking, or per-level allocation policy. **Do not expect identical keypoints or descriptors from the two implementations.**

It is **C++ source, not Python ORB**. The Python code invokes the scratch C++ executable on grayscale images, so it does not replace your implementation with OpenCV ORB. OpenCV is used as a separate reference extractor and as a common Hamming matcher/evaluation tool. The C++ `features::MatchBruteForce` implementation is also included as-is but isn't used in the benchmark (so descriptor matching is held constant between extractors).

## Contents

```text
orb_extractor_benchmark/
├── include/features/*.hpp     # original feature library headers
├── src/features/*.cpp        # original C++ features implementation, unchanged
├── src/orb_extract.cpp       # new standalone C++ PGM -> feature CSV adapter
├── CMakeLists.txt            # builds feature_core + orb_extract; C++17 only
├── bin/orb_extract           # precompiled Linux x86_64 (optional convenience)
├── benchmark.py              # dataset evaluator and OpenCV comparison
├── extract_image.py          # save scratch keypoints/descriptors for one image
├── run_all.py                # multi-dataset runner + combined summary
├── sample_data/bunny.png     # optional smoke test image from source repository
├── tests/test_metrics.py     # checks homography-based evaluation logic
├── requirements.txt
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

The sample image is a Stanford Bunny rendering from the original repository. Per the original notices, the Bunny imagery is restricted to research / non-commercial use and requires source attribution; do not treat it as MIT-licensed.

## 1. Setup (Ubuntu / Linux)

From inside the extracted ZIP:

```bash
cd orb_extractor_benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Recommended: build YOUR scratch extractor from source.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Benchmark scripts look for bin/orb_extract by default.
cp build/orb_extract bin/orb_extract
```

The binary in `bin/` is supplied for Linux x86_64 and may work without recompiling. If it fails on your system, build from source using the commands above. No Eigen, g2o, Sophus or C++ OpenCV dependency is required. CMake and a C++17 compiler are required for rebuilding.

### Quick test (included example)

```bash
python extract_image.py sample_data/bunny.png --output results/one_image
python benchmark.py synthetic --image sample_data/bunny.png --output results/synthetic \
  --levels 2 --features-per-level 150
```

Check `results/one_image/scratch_keypoints.jpg` and `scratch_features.csv`, and `results/synthetic/summary.csv`.

## 2. Run Oxford Affine

Expected extracted folder structure (Oxford Affine / Mikolajczyk benchmark):

```text
/datasets/oxford/
   bikes/img1.ppm, img2.ppm, ... img6.ppm, H1to2p, H1to3p, ...
   boat/img1.ppm, img2.ppm, ... img6.ppm, H1to2p, ...
   leuven/...
   graf/...
```

```bash
python benchmark.py oxford --root /datasets/oxford --output results/oxford --max-pairs 0
```

**With ground-truth homographies:** keypoint repeatability, correct matches, and matching precision. The `--max-pairs 0` switch processes **all** available image pairs; the default of 20 is a faster preview.

Oxford Affine dataset: https://www.robots.ox.ac.uk/~vgg/research/affine/

## 3. Run HPatches (full image sequences)

Expected structure:

```text
/datasets/hpatches-sequences-release/
   i_ajuntament/1.ppm, 2.ppm, ..., H_1_2, H_1_3, ...
   v_bark/1.ppm, 2.ppm, ..., H_1_2, ...
   ...
```

```bash
python benchmark.py hpatches --root /datasets/hpatches-sequences-release \
  --output results/hpatches --max-pairs 0
```

Use **HPatches full image sequences**, NOT the patch-only descriptor benchmark archive. Ground-truth metrics are available through each sequence's homographies.

HPatches: https://github.com/hpatches/hpatches-dataset

## 4. Run TUM RGB-D

Expected folder:

```text
/datasets/rgbd_dataset_freiburg1_xyz/
    rgb.txt
    rgb/1305031102.175304.png
    rgb/...
    depth/...
    groundtruth.txt
```

```bash
python benchmark.py tum --root /datasets/rgbd_dataset_freiburg1_xyz \
  --output results/tum_xyz --step 3 --max-pairs 100
```

A parent directory containing multiple `rgbd_dataset_*` folders is also supported. For ORB-only evaluation, only RGB frames are read; depth and trajectories are not needed. A **fundamental-matrix RANSAC inlier fraction estimated from matches** is reported. This is **not ground-truth matching precision** and does not establish pose accuracy.

TUM: https://cvg.cit.tum.de/data/datasets/rgbd-dataset

## 5. Run KITTI odometry

Expected folder:

```text
/datasets/kitti/dataset/sequences/
    00/image_0/000000.png, 000001.png, ...
    01/image_0/...
```

```bash
python benchmark.py kitti --root /datasets/kitti/dataset \
  --output results/kitti --step 3 --max-pairs 100
```

You can also pass a specific `sequences/00` directory. The benchmark reads grayscale `image_0` if available; otherwise it checks `image_2`. KITTI is evaluated with an **estimated F-RANSAC inlier fraction**, not ground-truth accuracy. Ground-truth trajectories aren't currently read by the benchmark.

KITTI: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

## 6. Any other image sequence

```bash
python benchmark.py folder --root /datasets/my_frames --output results/custom --step 1
```

Reads a sorted sequence of image files and compares neighbouring selected frames. Without homographies, it reports feature and matching statistics, plus estimated F-RANSAC inlier ratios (when possible).

## 7. Run all your installed datasets in one command

```bash
python run_all.py \
  --synthetic-image sample_data/bunny.png \
  --oxford /datasets/oxford \
  --hpatches /datasets/hpatches-sequences-release \
  --tum /datasets/rgbd_dataset_freiburg1_xyz \
  --kitti /datasets/kitti/dataset \
  --output results/all --max-pairs 0
```

Omit any dataset argument you haven't downloaded. Outputs are saved separately for each dataset, plus `results/all/all_datasets_summary.csv`.

## 8. Extract keypoints/descriptors from a single image

```bash
python extract_image.py /path/photo.jpg --output results/photo --extractor both
```

The outputs include `scratch_features.csv`, `scratch_keypoints.jpg`, and (with `--extractor both`) equivalent OpenCV files. The descriptor is 256 bits, encoded as **64 hex digits**. Keypoint positions are in the **original image's coordinates**. `angle_rad` is in radians; `scale` is the level-to-base scale factor. As in the source project, `--features-per-level` is a **per-pyramid-level** cap.

For direct C++ use on an 8-bit binary PGM image:

```bash
./bin/orb_extract --input image.pgm --output features.csv \
  --levels 8 --features-per-level 500 --fast-threshold 30
```

For regular PNG/JPEG/PPM inputs, use the Python wrapper; it converts them to PGM automatically. The C++ CLI does not rely on OpenCV.

## 9. Output explanation

Every `benchmark.py` call produces:

- **`pairs.csv`**: every evaluated pair, source image names, extractor, detected keypoints, matches, correct matches (where GT homography exists), matching precision, repeatability, times, and estimated F-RANSAC fraction for video data.
- **`summary.csv`**: per-extractor means across the evaluated pairs. These are **means of pair-level values**, not a globally pooled precision.
- **`settings.json`**: the exact benchmark parameters.
- **`plots/`**: comparison plots for key metrics (a plot is omitted when that metric isn't available).
- **`visualizations/`**: match images; for homography datasets, **green = geometrically correct**, **red = geometrically incorrect**. Only the first 80 matches are drawn per visualization.

For Oxford/HPatches/synthetic, a match is counted correct when its reprojection under the **known homography** is at most `--epsilon` pixels away (default: 3). Repeatability is the fraction of eligible source keypoints that have a distinct detection within the same tolerance. For synthetic warps, visibility masks are used to exclude points outside the warped source field of view.

For TUM and KITTI, matching precision and repeatability are **blank** because these scripts are not using per-pixel ground truth correspondences. The F-RANSAC fraction is only a noisy, model-dependent indicator (degenerate scenes, dynamic objects, and pure rotation can affect it).

### Important comparison caveats

1. Scratch ORB does **not** have an OpenCV-trained sampling pattern, Harris ranking, or OpenCV's octave budgeting. Same nominal settings **do not** guarantee same keypoint counts.
2. The total scratch keypoint budget is `levels × features-per-level`, unlike OpenCV's global `nfeatures`. For apples-to-apples comparisons, inspect actual detected counts, and tune settings if you want closer counts.
3. Matching is held constant: **mutual nearest-neighbour Hamming** via OpenCV for *both* extractors. That isolates feature quality from the matching algorithm, and differs from the original repository's one-way nearest match + distance threshold.
4. `extract_ms` measures extraction **inside each implementation**, excluding Python image decoding, PGM writing, subprocess startup and CSV parsing. Native vs Python-binding overhead can still differ.
5. Report correct-match count alongside precision: a method with 5/5 correct matches can have better precision but worse practical value than one with 100/120.
6. Results are not published Oxford or HPatches benchmark scores; they are metrics computed by this script with the definitions above. If preparing research results, report the settings and the correspondence definitions.

## 10. Options / troubleshooting

```bash
python benchmark.py --help
python -m unittest discover -s tests -v
```

Suggested smaller/faster debugging configuration:

```bash
python benchmark.py synthetic --image sample_data/bunny.png --output results/debug \
  --levels 2 --features-per-level 150 --max-pairs 3
```

Use `--visualizations 0` to avoid saving images, `--extractor scratch` to evaluate only your code, or `--extractor opencv` for a quick reference run. For video inputs, `--step 1` compares consecutive frames and `--step 3` compares frames 3 apart. If a sequence has fewer than `step+1` frames, it has no evaluable pairs.

**Licenses:** `LICENSE` and `THIRD_PARTY_NOTICES.md` copied from the source ZIP. Source code copyright notices and code contents preserved.

## 11. UPDATED: clean correspondence dashboard (like a two-column ORB report)

**New default visualization:** keypoints are small yellow dots rather than the large
scale/orientation rings. Display-only spatial sampling reduces clutter; the extractor,
all matches, accuracy metrics, and CSVs are **unmodified by visualization choices**.

From the project folder, run for example:

```bash
python benchmark.py synthetic --image sample_data/bunny.png \
  --output results/synthetic --levels 2 --features-per-level 150 \
  --max-pairs 3 --visualizations 3 --viz-max-matches 45
xdg-open results/synthetic/dashboards/index.html
```

This creates a **two-column HTML report** (OpenCV ORB on the left, original scratch C++
ORB on the right). Each column includes the original and second frames with their
correspondences, keypoint and match counts, inlier/outlier counts, times, and a
plain-language description of the method and geometry check. `dashboards/index.html`
links to each pair report; each pair also has a **standalone PNG report** in `dashboards/`. `visualizations/*.jpg` contains the clean side-by-side
match imagery without the table. Open the HTML using a regular browser; it works
offline if you keep `dashboards/` and `visualizations/` under the same output folder.

**Options to control clutter without affecting evaluation:**

```bash
# Display at most 25 correspondences; spread them out farther (display only).
python benchmark.py tum --root /path/to/rgbd_dataset_freiburg1_xyz \
  --output results/tum --step 3 --max-pairs 100 --visualizations 5 \
  --viz-max-matches 25 --viz-min-spacing 12 --viz-marker-radius 2
xdg-open results/tum/dashboards/index.html
```

`--visualizations 5` builds reports for the first 5 frame pairs, while `--max-pairs
100` processes **100 pairs** in the CSV metrics. `--viz-max-matches`,
`--viz-min-spacing`, and `--viz-marker-radius` affect the image **only**. The optional
`--max-hamming-distance 60` is different: it *does* filter matches before computing
matching metrics; by default it is 0, meaning no extra descriptor threshold.

To see keypoints alone without the overlapping rings:

```bash
python extract_image.py sample_data/bunny.png --extractor both \
  --output results/one_image --viz-max-keypoints 100 --viz-min-spacing 7
xdg-open results/one_image/scratch_keypoints.jpg
```

Both `scratch_features.csv` (ALL keypoints/descriptors) and a brief
`scratch_description.txt` are saved alongside the visualization.

**Important comparison note:** The original reference screenshot contains PnP
inliers and a ground-truth camera-pose error. This ORB-only benchmarking project
does not implement PnP or consume RGB-D depth: it intentionally reports **homography
GT correctness** for synthetic/Oxford/HPatches, and **estimated fundamental-matrix
RANSAC geometric consistency** for TUM/KITTI/folder. These quantities must not be
called PnP inliers or used as ground-truth pose accuracy. The CSV and HTML label
both cases explicitly.

---

## NEW: One HTML dashboard for EVERY dataset and image pair

This updated edition adds **`build_report.py`** and an automatic multi-dataset runner. Your existing
`datasets/` and `results/` directories remain on your laptop; the updated ZIP contains the code,
not downloaded datasets or your old experiment outputs.

### A. Add the new code without deleting any results

If you already have `~/Downloads/orb_extractor_benchmark`, go to Downloads and extract the new ZIP
**over your existing folder** (allow overwriting code files). Do **not** delete your existing project.
Your `datasets/` and `results/` folders will remain intact.

```bash
cd ~/Downloads
unzip -o ORB_Extractor_Benchmark_Multi_Dataset.zip
cd orb_extractor_benchmark
source .venv/bin/activate
```

### B. Immediately create a master page from EXISTING results (no benchmarking required)

```bash
python build_report.py --results results
xdg-open results/index.html
```

The master page groups results under **Oxford Affine**, **HPatches**, **TUM RGB-D**,
**KITTI Odometry**, **Synthetic**, and **Custom** headings. Each contains expandable sequences
and clickable image-pair cards with small OpenCV / Scratch previews where available.
Click a card to open the familiar **two-column correspondence comparison** and tables.
The master page has a search box to find a sequence or image.

**Important:** Older benchmark runs made with `--visualizations 10` only saved 10 pairs'
match images. The master report **will include all rows of `pairs.csv`**, but for pairs without
saved correspondence images it shows a **clearly marked metrics-only comparison**. It cannot
reconstruct missing match lines from CSV. To generate real comparisons for **every pair**, do C.

### C. Run ALL available datasets and save an image for EVERY evaluated pair

The automatic runner recognizes the directory layout
`datasets/{oxford,hpatches,tum,kitti,custom}` and skips missing datasets.

```bash
python run_all.py \
  --auto-datasets datasets \
  --output results \
  --levels 8 \
  --features-per-level 150 \
  --fast-threshold 30 \
  --max-pairs 0 \
  --step 3 \
  --visualizations -1 \
  --viz-max-matches 30 \
  --viz-min-spacing 12 \
  --skip-dashboard-pngs
```

- `--max-pairs 0`: **evaluate every available image pair** per dataset.
- `--visualizations -1`: **save two JPEG match images + an HTML comparison for every pair**.
- `--viz-max-matches 30`: draw at most 30 lines **per image pair**, purely for readability.
  **All matches are still used in metrics**.
- `--skip-dashboard-pngs`: skip a large optional PNG screenshot for each comparison to save disk
  space. The side-by-side HTML pages and JPEG visualizations are still generated.
- `--step 3`: for TUM, KITTI, and custom video sequences, match frame `i` to frame `i+3`.
  The benchmark does **not** evaluate all possible N² combinations of frames.

**Warning:** Entire TUM or KITTI sequences can yield thousands of pairs and occupy substantial
storage and compute time with full visualizations. For a trial run, set `--max-pairs 10`.
The runner automatically regenerates `results/index.html` at the end.

Open it:

```bash
xdg-open ~/Downloads/orb_extractor_benchmark/results/index.html
```

You can also run individual datasets and rebuild the master page:

```bash
python benchmark.py hpatches \
  --root datasets/hpatches/hpatches-sequences-release \
  --output results/hpatches \
  --max-pairs 0 --visualizations -1 --skip-dashboard-pngs
python build_report.py --results results
xdg-open results/index.html
```

### D. Data interpretation

For Oxford / HPatches / synthetic, a blue correspondence is **ground-truth homography correct**.
For TUM / KITTI / custom video, blue marks **estimated fundamental-matrix RANSAC consistency**;
it is **not ground-truth precision or PnP accuracy**. The comparison uses the *same mutual Hamming
matcher* for both extractors. The retained original C++ ORB feature code has not been modified.
