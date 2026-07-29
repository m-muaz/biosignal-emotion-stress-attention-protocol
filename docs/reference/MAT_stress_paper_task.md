# Summary & Protocol Breakdown

## What is the Trier Social Stress Test (TSST)?

The **Trier Social Stress Test** is a standardized, lab-based protocol designed to reliably induce psychological stress in participants. It combines two potent stressors:

1. **Social-evaluative threat** — being watched, judged, or evaluated by others (a panel, a video camera, or even a leaderboard with "top scorers")
2. **Uncontrollability** — tasks that are hard to perform correctly under time pressure

The classic TSTS involves a **public speaking task** (mock job interview) followed by a **mental arithmetic task** (serial subtraction out loud in front of an audience). This paper adapts the TSST by keeping only the mental arithmetic component and replacing the live panel with a web interface that uses timers, dummy leaderboards, and "Hurry Up!" messages to create stress digitally — making it far easier to deploy outside a lab.

## High-Level Summary

The authors investigate whether **heart rate (HR) extracted from ECG signals** can reliably reflect stress induced by mental arithmetic of escalating difficulty. They build a web-based Mental Arithmetic Task (MAT) interface with four levels of increasing complexity and conduct a pilot with 8 participants wearing Shimmer clinical-grade ECG sensors. Their key finding: mean HR rises during the MAT phases (peak of 84.43 BPM at Level 1) compared to baseline (77.15 BPM) and drops lowest during the post-task resting phase (75.66 BPM), supporting HR as a real-time stress indicator.

---

## Protocol Design — Full Walkthrough for Replication

The study follows a strict linear sequence (shown in Figure 1 of the paper):

### Phase 1: Consent & Baseline (3 min)
- Landing page with a consent button
- A **3-minute baseline** period while **relaxation music** plays (they link a YouTube embed)
- Participant sits quietly, sensors recording

### Phase 2: Instructions & Demo Trial
- Instruction page explaining the quiz flow
- A **demo trial** so the participant knows what to expect — important to avoid surprise confounds

### Phase 3: Mental Arithmetic Task — 4 Levels

| Level | Operations | Numbers | Time per Q | Total Time | Example |
|-------|-----------|---------|------------|------------|---------|
| L1 | +, − | 2 | 4 sec | 120 sec | `9 + 7` |
| L2 | +, −, × | 3 | 4 sec | 120 sec | `8 − 4 × 2` |
| L3 | +, −, × | 4 | 5 sec | 150 sec | `7 × 4 + 6 − 21` |
| L4 | +, −, ×, ÷ | 4 | 5 sec | 150 sec | `72 / 3 + 9 − 5` |

Each level has **30 questions**. Between levels, participants stay at their desk for a short break but cannot leave.

**Stress-amplifying elements built into the UI** [Study Design](https://www.alphaxiv.org/abs/2403.10356?page=3):
- A **dummy leaderboard** showing fake top-5 names and scores (social comparison / evaluative threat)
- A **"Hurry Up"** message that starts blinking when half the question time remains (time pressure)
- Questions get progressively harder while time pressure stays tight

### Phase 4: Resting Period
- Relaxation music plays again
- Post-task recovery phase serves as a physiological baseline comparison

### Sensor Setup
They used **Shimmer** wearable sensors (ECG, PPG, and EDA — clinical grade) [Pilot Study](https://www.alphaxiv.org/abs/2403.10356?page=3). In this paper they analyze only the **ECG** data:
- Filtering and R-peak detection
- Heart rate computed via **Neurokit2** (Python package) [Analysis](https://www.alphaxiv.org/abs/2403.10356?page=4)

---

## What Made Stress Detectable

The mean HR values tell the story clearly:

| Phase | Mean HR (BPM) |
|-------|:------------:|
| Baseline | **77.15** |
| L1 (easiest) | **84.43** (peak) |
| L2 | **83.17** |
| L3 | **81.78** |
| L4 (hardest) | **81.62** |
| Resting | **75.66** (lowest) |

The interesting pattern is that **Level 1**, not the hardest level, produced the highest HR. The authors don't speculate at length, but one plausible explanation: L1 is the first encounter with time pressure and evaluation, so the **novelty + anticipation** spike outweighs later habituation — a useful design consideration if you're building your own protocol.

---

## Key Takeaways for Recreating This

| Element | What Matters |
|---------|-------------|
| **Time pressure** | Tight per-question deadlines (4–5 sec) are the primary stressor |
| **Social evaluation** | A dummy leaderboard is cheap and effective — no human panel needed |
| **Interruption cues** | Blinking "Hurry Up" halfway through each question |
| **Progression** | Increasing difficulty keeps stress from dropping via boredom |
| **Pre/post baselines** | Relaxation music before and after gives clean comparison windows |
| **Duration** | ~7 min baseline + 2 min demo + ~9 min MAT + ~3 min rest = ~21 min total session |
| **Screen-based** | Entirely web, works on phones and laptops — no special lab hardware besides the wearable |

The web interface itself is [publicly accessible](https://mjyadav7.github.io/MAT_QUIZ.github.io/) — though it's the participant-facing quiz, not the admin dashboard. Still, the structure gives you something to fork or adapt rather than building from scratch.