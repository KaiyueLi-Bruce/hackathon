# ChromaLog

A macOS app for TLC (Thin-Layer Chromatography) plate analysis. Drop in a photo, get Rf values, AI interpretation, and a searchable experiment archive — automatically.

---

## What it does

TLC is a daily routine in organic chemistry labs: develop a plate, hold it under UV, manually measure distances, hand-calculate Rf. ChromaLog automates that process:

1. **Import** a TLC plate photo
2. **Auto-detect** baseline, solvent front, and spots (OpenCV pipeline)
3. **Calculate Rf** values instantly
4. **Generate an AI report** — reaction status, spot interpretation, next-step suggestions
5. **Save** to a searchable local archive

Manual adjustment is always available — drag lines and spots to correct anything the auto-detection got wrong.

---

## Requirements

- macOS 14 (Sonoma) or later
- Python 3.10+ (for the CV sidecar)
- An [Anthropic API key](https://console.anthropic.com/) (for AI reports)
- Xcode 15+ (to build from source)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/KaiyueLi-Bruce/hackathon.git
cd hackathon
```

### 2. Set up the Python sidecar

```bash
cd cv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the sidecar

```bash
# From the cv/ directory, with .venv activated
python run.py
```

The sidecar runs at `http://localhost:8765` and handles all image processing. Keep this terminal open while using the app.

### 4. Build and run the app

```bash
cd ../App
swift build   # or open in Xcode and press ▶
```

Or open `App/` as a Swift package in Xcode and run the `ChromaLog` target.

### 5. Add your API key

In the app: click the **gear icon** (top-right) → paste your Anthropic API key → Save.

---

## Basic workflow

### Analyze a plate

1. **Import a photo** — drag and drop a TLC plate image onto the canvas, or click **Import** in the floating toolbar.
2. **Auto-detect** — click the **Auto-detect** button (highlighted in the toolbar). The sidecar detects:
   - Baseline and solvent front lines
   - All spots with their Rf values
   - Lane assignments
3. **Review** — Rf values appear in the **Results** tab on the right. Drag the baseline or solvent front lines to fine-tune if needed. Click spots to add labels (SM / Product / By-product / Standard).
4. **Generate AI report** — switch to the **AI** tab → click **Generate AI report**. The report covers:
   - Reaction status (complete / incomplete / inconclusive)
   - Spot-by-spot interpretation
   - Next-step suggestions
5. **Save** — press **⌘S** or click the save button. The plate is stored in the local archive with its photo, Rf data, and report.

### Browse the archive

Click the **grid icon** in the left rail to open the archive. Search by experiment name, date, or Rf range.

---

## Teaching the detector (online learning)

The spot detector improves as you correct it:

1. Run auto-detect on a plate.
2. **Add missed spots** by clicking on the plate. **Remove false positives** by double-clicking a spot.
3. Save the plate — corrections are immediately fed back to the classifier.

After a few plates the detector learns your typical plate appearance and needs fewer corrections. The inspector shows **"Learned from N corrections"** to track progress.

---

## YOLO model (optional, higher accuracy)

A YOLOv8-based detector is available as a higher-accuracy fallback. It activates automatically once trained.

### Train the YOLO model

```bash
cd cv
source .venv/bin/activate
pip install ultralytics   # one-time
python train_yolo.py --epochs 50 --n-synth 2000
```

Training takes ~60–90 minutes on Apple Silicon (MPS). The script:
1. Generates 2000 synthetic TLC images from photos in `training_pictures/`
2. Trains YOLOv8n for 50 epochs
3. Exports to `cv/models/yolo_spot.onnx`

You can also trigger training from the app: **Settings → YOLO Spot Detector → Re-train**.

Once the model is ready, the status dot turns green and YOLO is used automatically when the standard detector finds zero spots.

---

## Project structure

```
hackathon/
├── App/                    # SwiftUI macOS app (Swift Package)
│   └── Sources/ChromaLog/
│       ├── AppStore.swift  # Central state
│       ├── CVClient.swift  # HTTP client for the sidecar
│       └── Views/          # UI components
├── cv/                     # Python sidecar (FastAPI + OpenCV)
│   ├── chromalog_cv/       # Detection pipeline
│   │   ├── pipeline.py     # Main pipeline entry point
│   │   ├── spots.py        # Spot detection & Rf calculation
│   │   ├── rectify.py      # Perspective correction
│   │   ├── learn.py        # Online incremental classifier (SGD)
│   │   └── yolo.py         # YOLO ONNX inference
│   ├── train_yolo.py       # YOLO training script
│   ├── models/             # ONNX model files (gitignored)
│   └── tests/              # pytest test suite (45 tests)
├── training_pictures/      # Real TLC photos used for YOLO training
└── docs/                   # Design specs
```

---

## Detection pipeline

```
Photo
 → Perspective correction (OpenCV contour → homography)
 → CLAHE illumination normalization
 → Auto-polarity binarization (minority class = spots)
 → Hough line detection (baseline + solvent front)
 → Connected-component spot candidates
 → Lane assignment (x-projection histogram)
 → SGD patch classifier (if trained, improves with corrections)
 → YOLO fallback (if model exists and classifier finds 0 spots)
 → Rf = (baselineY − spotY) / (baselineY − frontY)
```

---

## Running tests

```bash
cd cv
source .venv/bin/activate
pytest tests/ -q
```

45 tests covering detection, learning, YOLO inference, and the FastAPI endpoints.

---

## Keyboard shortcuts

| Action | Shortcut |
|--------|----------|
| Save plate | ⌘S |
| Toggle left rail | toolbar sidebar button |
| Toggle inspector | toolbar right-sidebar button |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| macOS UI | SwiftUI (macOS 14+) |
| Local storage | SQLite via GRDB.swift |
| Image processing | Python · OpenCV · FastAPI |
| Spot classification | scikit-learn SGDClassifier (online learning) |
| YOLO detection | Ultralytics YOLOv8n → ONNX Runtime |
| AI reports | Anthropic Claude API |
