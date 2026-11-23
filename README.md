# EyeTrackTrial

Real-time gaze tracking from a close-up eye camera with
[VOGeo-Gaze](https://github.com/DSGZ-MotionLab/VOGeo-Gaze) (MICCAI 2026), plus a calibration and an
accuracy trial that measures where on the screen the system thinks the participant is looking.
The GUI follows the SmartAnkle launcher (same theme, Italian labels).

```
python app.py
```

## Install

Python 3.10+. EyeTrackTrial uses the `vogeo_gaze` package (model and weights) from the
VOGeo-Gaze repository.

**NVIDIA GPU (recommended).** Install the CUDA build of PyTorch first, then the rest.
Pick the CUDA version your driver supports (`nvidia-smi`, top right); `cu121` and `cu124` work
on any recent driver:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/DSGZ-MotionLab/VOGeo-Gaze.git
pip install -r requirements.txt
```

The folder also works as a subfolder of a VOGeo-Gaze checkout (`VOGeo-Gaze/EyeTrackTrial/`). It then
uses the `vogeo_gaze` next to it, with no extra install.

Check that the GPU is visible:

```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**CPU only.** Same, with `--index-url https://download.pytorch.org/whl/cpu`. The model then runs at
about 10–15 frames/s on a laptop CPU, enough to try the program but slow for saccades.

In the **Live** tab, *Dispositivo* lists every NVIDIA GPU torch can see (`GPU 0 · <name>`),
*Automatico* (GPU when available) and *CPU*. *Mezza precisione fp16* runs the network in half
precision on the GPU (faster on RTX cards). The *GPU NVIDIA* chip in the header turns red when the
installed PyTorch is CPU-only.

## Live tab

- **Ingresso**: every camera Windows reports (names via DirectShow), a recorded eye video
  (*File video...*, played in real time, looping) or *Simulatore occhio (test)*, a synthetic eye
  that looks at the trial targets, to try the whole program without a camera.
- **Risoluzione**, mirror / flip: the frame is converted to gray and letterboxed into the
  model's 320×240 canvas, as in the offline tool.
- The camera view shows the detected pupil (white), the refraction-corrected pupil (yellow) and the
  iris (cyan); the segmentation view shows pupil (blue), iris (red) and open eye (green); below,
  the last 10 s of θ horizontal / vertical.

The model expects a **close-up eye camera** (one eye filling the frame, as in VOG goggles), the
kind of images it was trained on.

### How real time works

Each new frame goes through the per-frame part of the network (segmentation, ellipse fits,
refraction correction, closed-form eye reconstruction). The newest frame is always processed and
older ones are dropped, so latency does not build up when the model is slower than the camera.
F_tem, the temporal stage, runs in the background on the last 16 frames sampled at 12 Hz and
updates the **eyeball centre** (median of the recent clip centres). Every frame's gaze is its
pupil ray lifted onto the eyeball sphere about that centre, as in the offline tool.

The centre needs a few seconds of eye movement to settle (*Centro occhio: stimato*). It is
**frozen at the start of calibration** and stays frozen for the experiment, so the calibration
mapping remains valid. *Reimposta centro occhio* starts over and invalidates the calibration.

## Prova tab

1. **Calibrazione**: 5, 9, 13 (default), 17 or 25 points on the participant screen, centre first,
   the rest shuffled. Each dot shrinks for 0.6 s and is then held for *Durata per punto*. The
   robust median of the valid samples of every point (blinks and low-confidence frames dropped)
   is mapped to the screen with a second-order polynomial in (θH, θV) (affine with fewer than 8
   points). The tab shows the mean / max residual and the leave-one-out error, and draws each
   point with its fitted gaze. From then on, the red dot on the map is the live gaze.
2. **Esperimento**: the selected target types, shuffled:

   | Type | Stimulus | Error measured |
   |---|---|---|
   | Punto piccolo / Punto grande | dot (r = 5 / 18 px) for *Durata fissazione* | from 0.5 s after onset |
   | Saccade | central cross 0.8–1.2 s, then a jump to a peripheral dot for 1.2 s | from 0.45 s after the jump |
   | Inseguimento | dot moving on an ellipse (5–7 s per turn) | from 0.6 s, against the dot position at each sample |
   | Lettera | a letter (34 pt) | from 0.5 s |
   | Immagine | a picture from `EyeTrackTrial/stimuli/` (PNG/JPG, 80 px) or a built-in icon | from 0.6 s |

   Static targets are spread evenly over the 3×3 screen regions (saccade targets avoid the centre).

The **participant screen** can be any monitor (a second monitor by default). With one monitor the
trial runs fullscreen over the GUI. On the participant screen, `Space` starts, `Esc` aborts.

### Blinks (ammiccamenti)

- A frame counts as **closed** when the model's open-eye score (share of the iris inside the open
  eye) is below 0.7, the same threshold as the offline tool, or when the pupil is not found
  (confidence < 0.15). A run of closed frames is one blink: 40–600 ms is an *ammiccamento*, longer is a
  *chiusura*, shorter is a *perdita breve* (dropout).
- The eyelid covers the top of the pupil before the score drops, and the eye makes a small movement
  on reopening, so samples from **100 ms before to 150 ms after** each blink are excluded too.
- Blink frames are not fed to F_tem, so they don't pull the eyeball centre.
- **Live**: the camera view gets an orange border labelled *OCCHIO CHIUSO*, *Ammiccamenti* shows the
  count and the rate per minute, and closed periods are shaded grey in the gaze trace.
- **Repeats**: a calibration point or trial whose analysis window keeps less than 50 % valid samples
  (or fewer than 5) is replayed at the end of the sequence, in the same window, up to *n* times
  (*AMMICCAMENTI* card, default 2). Calibration pools a point's samples over its attempts. The
  experiment analyses each trial's last attempt.
- **Output**: per trial and per category, *Persi blink* (share of samples lost to blinks),
  *Ammicc.* (blinks during the trial) and *Ripetute* (trials replayed). `ammiccamenti.csv` and
  `calibrazione_ammiccamenti.csv` list every blink (start, end, duration, type, trial), and
  `riepilogo_ammiccamenti.json` has the count, rate per minute and median duration.

### Errors

All errors are in **screen pixels**. A trial's error is the median, over its valid samples in
the analysis window, of the distance between the estimated gaze and the target. Smooth pursuit is
cut into 0.5 s segments, each counted as one trial at its mean position. The tables give, per
**target type**, **screen region** (3×3) and **eccentricity** (4 bands of distance from the screen
centre): trials, valid trials, median / mean / SD / Q1–Q3 error, precision (RMS of sample-to-sample
gaze differences) and the share of valid samples.

Pursuit error includes the system latency (camera + model) times the target speed. Static
targets are measured after the eye has settled, so they show the spatial accuracy.

### Output

`EyeTrackTrial/risultati/<subject>_<date>_<time>/`:

| File | Content |
|---|---|
| `calibrazione.json` | mapping, per-point residuals, frozen eyeball centre, device, settings |
| `calibrazione_campioni.csv` | every gaze sample recorded during calibration |
| `esperimento_campioni.csv` | every sample of the experiment, with mapped gaze, target and error |
| `prove.csv` | one row per trial (or pursuit segment) |
| `riepilogo_tipo.csv`, `riepilogo_regione.csv`, `riepilogo_eccentricita.csv` | the three tables |
| `eventi.json` | stimulus onsets / offsets (`time.perf_counter`, the clock the frames are stamped with) |

## Files

| File | Role |
|---|---|
| `app.py` | GUI |
| `tracker.py` | single-frame VOGeo-Gaze, running eyeball centre, device selection (CUDA / CPU) |
| `sources.py` | cameras, video files, eye simulator (newest-frame grab threads) |
| `stimuli.py` | participant screen and trial sequences |
| `calibration.py` | calibration grids and the polynomial mapping |
| `analysis.py` | per-trial and per-category errors |
| `blinks.py` | blink detection, margins, live counter |
| `theme.py` | SmartAnkle palette and ttk styles |
