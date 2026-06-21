# Auto Chem (macOS)

TLC plate analysis app — SwiftUI, canvas-first. Built per `../ChromaLog-Spec.md`.

## Run

```bash
cd App
swift run            # builds and launches the app
# or
swift build && ./.build/debug/ChromaLog
```

Requires the Xcode command-line toolchain (Swift 5.9+). The project is a Swift
Package executable so it builds and runs from the CLI; source files are
organized to drop into an Xcode app target later.

## Try the demo flow

1. Drag a TLC photo from `../training_pictures/` onto the canvas (or use **Import**).
2. Drag the **Solvent front** and **Baseline** reference lines to the plate's
   real top/origin.
3. Click **Spot**, pick a label chip (SM / Product / By-product / …), then tap
   each spot on the plate. Drag a marker to fine-tune; relabel or delete from
   the **Results** panel.
4. Read live **Rf values** in the inspector. **Co-spot check** flags aligned
   spots (ΔRf < 0.05). Fill in **Conditions** (solvent, ratio, …).
5. Press **Redraw** in the toolbar to see the standardized digital plate.
6. Press **Save** (`⌘S`). Open the **Library** (grid icon, or rail) to search
   and reload saved plates.
7. Toggle panels: `⌘[` rail · `⌘]` inspector · `⇧⌘F` focus mode.

Saved data lives in `~/Library/Application Support/ChromaLog/` (SQLite + images).

## Milestones

- **M0** — canvas-first shell, drag-drop import, floating toolbar, filmstrip,
  English UI, system light/dark. ✅
- **M1** — manual calibration, spot picking, Rf calculation + live Rf table. ✅
- **M2** — digital plate redraw + GRDB/SQLite persistence + searchable archive. ✅
- **M3** — Anthropic API single-plate report. _next — needs your API key_
- **M4** — UI polish, label picker, dark-mode-safe styling. ✅
- **M5** — Python sidecar + OpenCV/ONNX auto-detect (manual stays as fallback).
  _needs local Python env_
