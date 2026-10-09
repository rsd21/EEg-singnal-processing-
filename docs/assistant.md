# EEG analysis assistant

`eegproc chat` starts a conversation with Claude, Anthropic's AI model, about your recording.
The assistant does not guess a pipeline. It first asks what kind of study you are doing and what
you want to find out, then proposes a plan, runs each step with the eegproc toolkit, shows you
the figures, and asks for your view before any decision that changes the data: which channels
to mark bad, which ICA components to remove, which filter cut-offs to use.

```
you> I recorded 2 minutes of resting EEG with blinks. I want blink rate and shape,
     and clean data for an alpha analysis afterwards.

  Checking quality first, then detecting the blinks.
  -> assess_quality()
     Overall score 99/100. Suggested bad channels: none
  -> detect_blinks()
     Channel VEOG: 27 blinks in 2.0 min = 13.5 blinks/min

All 32 channels look good. You blinked 13.5 times per minute (median width 391 ms,
median amplitude 161 µV). For the alpha analysis I suggest removing the blinks with ICA
rather than cutting data out. Shall I fit ICA and show you the components?
```

## Install and set up

```bash
pip install "eegproc[assistant] @ git+https://github.com/rsd21/EEg-singnal-processing-.git"
export ANTHROPIC_API_KEY=sk-ant-...      # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

The assistant needs Python 3.10 or newer and an Anthropic API key from
[console.anthropic.com](https://console.anthropic.com). API use is paid per token; when you type
`quit` the session prints how many tokens it used.

## Run it

**Command line**

```bash
eegproc chat recording.edf          # any supported format
eegproc chat demo:blinks            # simulated data: demo:resting, demo:oddball, demo:motor_imagery
eegproc chat recording.vhdr --effort high -o results/
```

**Python or Jupyter**

```python
from eegproc.assistant import chat

assistant = chat("recording.edf")
```

**Google Colab:** open
[notebooks/eeg_assistant_colab.ipynb](https://colab.research.google.com/github/rsd21/EEg-singnal-processing-/blob/main/notebooks/eeg_assistant_colab.ipynb).
It reads the API key from Colab's Secrets panel and shows figures inline.

Type `quit` to finish. Figures, saved data, feature tables and reports go to
`eeg_assistant_output/` (or the folder given with `-o`).

## What it can do

The assistant works through 29 tools that wrap the toolkit:

| Area | Tools |
|---|---|
| Data | load any supported file or a demo, describe the current state, fix channel types, drop channels |
| Quality | 0-100 score per channel with reasons, mark/unmark bad channels |
| Cleaning | band-pass/high-pass/low-pass, notch (auto 50/60 Hz), re-reference (average, chosen channels, bipolar, Laplacian), interpolate bad channels, resample, crop, ICA with eye/heart/muscle labels, mark artifact segments, undo |
| Blinks | detection with rate, width, amplitude, regularity and blink-shape figure; blinks as events for blink-locked epochs |
| Spectra | band power per channel, peak alpha frequency, 1/f exponent, scalp maps, connectivity (coherence, imaginary coherence, PLV, PLI, wPLI, envelope correlation) |
| Events | list events, epochs around events or fixed-length segments, ERPs with peak latency and amplitude, ERD/ERS |
| Channel performance | rank channels by quality, SNR, band power, connectivity, ERP SNR, ERD, discriminability, decoding accuracy |
| Output | figures, feature tables (CSV), cleaned data in any writable format, HTML report |
| Custom | `run_python` for anything the tools don't cover; you see the code and approve every run (turn off with `--no-python`) |

## What is sent to Anthropic, and what is not

- **Sent:** your messages, and what the tools return: summaries, tables and figures.
- **Not sent:** the raw signals. Patient fields (name, ID, birth date) and the recording date from
  file headers are left out of every summary.
- Files the assistant writes stay in the output folder; it cannot write elsewhere.

Check your institution's rules before using it with patient data.

## Settings

| Option | CLI | Python | Default |
|---|---|---|---|
| Model | `--model` | `model=` | `claude-opus-5-5` |
| Thinking effort | `--effort low/medium/high/xhigh/max` | `effort=` | `medium` |
| Output folder | `-o` | `output_dir=` | `eeg_assistant_output` |
| Python tool | `--no-python` | `allow_python=False` | on, with approval |
| Show figures to Claude | | `send_figures=False` | on |

Higher effort makes the model think longer (slower, more tokens). `send_figures=False` saves
tokens, but then Claude cannot read the figures, only the numbers. After about 20 MB of figures
in one session, new figures are no longer sent to Claude (you still see them), because every
request carries the whole conversation and the API limits a request to 32 MB.

The request settings (adaptive thinking with progress notes, server-side fallback) are written
for current models such as `claude-opus-5-5` and `claude-sonnet-5-5`; older models may reject
them.

Technical details: the assistant streams its replies, caches the system prompt and conversation
to reduce cost, shows short progress notes between tool calls, and enables Anthropic's
server-side fallback, which retries on another Claude model if a request is declined by a
safety classifier. The fallback has a separate rate limit.

## Limits

- **Offline only:** it analyses recordings, not live streams, so it cannot drive neurofeedback
  or BCI control in real time.
- **No sleep staging or seizure detection:** it can show spectra and spectrograms but does not
  classify them.
- **Mistakes:** the assistant can choose a poor setting or misread a figure. It reports numbers
  only from tool output and stops for your decision at each judgment call; check its steps,
  and use undo if needed.
- **Not clinical:** this is a research and teaching tool, not clinically validated. It will not
  diagnose.
