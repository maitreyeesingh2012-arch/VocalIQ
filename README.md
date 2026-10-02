# VocalIQ

A personal AI vocal coach. Sing (or upload a take), and VocalIQ measures pitch, breath,
vibrato, resonance, registers and dynamics, then coaches you with personalised tips, a
7-day practice plan and live practice tools.

Runs as a website, installs as a web app (PWA), and is set up to ship to the App Store and
Google Play with Capacitor. See **[MOBILE.md](MOBILE.md)** and **[DEPLOYMENT.md](DEPLOYMENT.md)**.


## Measuring results

```powershell
.\.venv\Scripts\python.exe scripts\usage_report.py --since 2026-10-09 --out docs\usage.md   # users, sessions, before/after scores
.\.venv\Scripts\python.exe scripts\pitch_accuracy.py --out docs\pitch_accuracy.md            # pitch error in cents (synthetic)
.\.venv\Scripts\python.exe scripts\pitch_accuracy.py --csv my_notes.csv                      # your own recordings of known notes
```

See `docs/` for the trial survey, teacher rating sheet and dev log template.

## Run it locally (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe app.py          # http://127.0.0.1:5000
.\.venv\Scripts\python.exe app.py --dev    # auto-reload while developing
```

Run the tests (about 20 seconds; they analyse synthetic singing):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

The command-line coach still works on its own: `python vocal_analyzer.py --help`.

## Features

| | Feature | Where |
|---|---|---|
| 1 | Session charts: pitch map by register, loudness, vibrato gauges, each with a table view | `web/js/charts.js`, `views/sessions.js` |
| 2 | Sing along with a song: finds your part of the original, compares melody and timing, marks the worst moments | `ReferenceSongAnalyzer` in `vocal_analyzer.py` |
| 3 | Progress: streaks, practice calendar, score trends per area, range growth | `views/progress.js`, `/api/progress` |
| 4 | Singer profile questionnaire that personalises tips | `views/profile.js`, `/api/profile` |
| 5 | Replay recordings and A/B compare with another take | `views/sessions.js`, `/api/session/<id>/audio` |
| 6 | Live pitch meter (tuner) with "hold this note" goals | `views/tools/tuner.js` |
| 7 | Guided warm-ups on piano with call-and-response scoring | `views/tools/warmup.js` |
| 8 | Pitch-matching drills that revisit your missed notes | `views/tools/match.js` |
| 9 | Range finder on a piano keyboard | `views/tools/range.js` |
| 10 | Breath trainer (timed "sss" with a steadiness score) | `views/tools/breath.js` |
| 11 | 7-day practice plan built from your weakest areas, plus daily reminders in the apps | `server/plans.py`, `views/today.js`, `js/reminders.js` |
| 12 | Share a read-only report (optionally with audio) with a teacher; save as PDF | `server/share.py`, `web/share.html` |
| 13 | Click the pitch map to replay any phrase | `PitchChart` in `charts.js` |

Accounts: email/password, or **guest** (one click; can be upgraded later without losing
anything). Account deletion is built in (an App Store requirement).

## Project layout

```
app.py                 entry point (waitress server; --dev for Flask's reloader)
vocal_analyzer.py      the analysis engine (also a standalone CLI)
server/                Flask API
  config.py            settings from environment variables
  db.py                SQLite schema and migrations (older databases upgrade in place)
  auth.py              cookie + bearer-token auth, guests, CORS, profile, account deletion
  analysis.py          background analysis jobs, audio storage, personalisation
  sessions.py          history, reports, audio, progress
  practice.py          practice results, range/drill summaries, weekly plan
  plans.py             7-day plan generator
  share.py             teacher share links
web/                   the app UI: plain HTML/CSS/ES modules, no build step
  index.html           landing page + app shell      share.html  public report page
  css/ js/ views/      app code                       sw.js       offline shell (PWA)
mobile/                Capacitor config, icons, and the script that bundles web/ for the apps
tests/                 API + analysis tests
data/                  recordings and song cache (created at runtime; not in git)
```

## How it fits together

- The browser records WAV itself (Web Audio), so the server never needs browser codecs.
- `POST /api/analyze` returns a job id; the client polls `/api/jobs/<id>` for progress
  messages. Jobs run in a thread pool inside the server process.
- Each session stores a compact chart series (`session_series`) plus a 22 kHz MP3 of the
  cleaned take, so reports and replay work without re-analysing.
- Live practice tools run entirely on the device (YIN pitch detection in `js/audio.js`);
  only their results are sent to the server.

## Charts

Series colours are the validated categorical slots 1–3 (chest / mix / head), checked
all-pairs against this app's own card surfaces (`#fffdf8` light, `#17131e` dark): CVD ΔE ≥ 9.2,
normal-vision ΔE ≥ 20.9. The light-mode aqua sits below 3:1 contrast, so every chart ships a
text legend and a table view. Status colours (good/warn/focus) always appear with a label.
