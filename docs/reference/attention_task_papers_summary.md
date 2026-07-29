Let me start by looking up the specific references you mentioned and then search broadly for BCI attention/focus dataset protocols.

Found the MEMA dataset paper — it's the one at ID 2411.09879. Let me now read the most relevant papers in parallel to extract their protocols and also search for non-EEG modality attention datasets.

Excellent — I've got rich detail on several key datasets. Let me also check for clinical trial BCI attention work and a few more specific datasets.

Here is a comprehensive breakdown of the landscape — organized by the protocol patterns you would need to replicate when collecting your own biosignal datasets for attention/focus BCI tasks.

---

## The Core Protocol Architectures for Inducing Attention in BCI

Every attention-induction protocol in this space works through a common logic: **establish a baseline, impose a cognitive demand that requires sustained focus, then validate that focus through behavioral performance**. What differs across datasets is which side of attention they target (sustained vs. selective, internally-generated vs. externally-cued) and which modalities they record. Here are the major protocol families:

---

### 1. The Three-State Passive Induction Protocol

This is the cleanest design for collecting labeled attention data. Used by the **[MEMA dataset](https://www.alphaxiv.org/abs/2411.09879)** (Liu et al., 2024) — the one you referenced.

**Attention states induced:**
- **Neutral** (1 min): Blank video, subject maintains normal state — neither relaxing nor concentrating.
- **Relaxing** (5 min): Soothing scenery video with relaxing music — subjects instructed to "stay as relaxed as possible."
- **Concentrating** (5 min): Machine learning course lecture video — subjects required to "concentrate, actively learn, and think," then answer a multiple-choice question afterward to validate focus. [Task Design](https://www.alphaxiv.org/abs/2411.09879v2?page=3)

**Validation mechanism:** Post-trial self-assessment + comprehension quiz for concentrating condition.

**Recording specs:** 20 subjects, 12 trials each (4 rounds × 3 state types, randomized order), 32-channel EEG at 500 Hz (ZhenTec-NT1-32, 10-10 system). Also collected emotion labels (VAD model) and Big Five personality traits. [Data Acquisition](https://www.alphaxiv.org/abs/2411.09879v2?page=2)

**Key design choice for your dataset:** Each trial: 5s hint → video clip (1-6 min) → 15s self-assessment → 15s rest. Task order within each round is randomized to avoid sequence effects. [Overall Procedure](https://www.alphaxiv.org/abs/2411.09879v2?page=2)

---

### 2. The Active Cognitive Load Manipulation Protocol

Used when you want graded, parametrically-controlled attention demands rather than categorical states. The classic tool here is the **n-back task**, which can be administered in auditory or visual form.

**From the [fNIRS + Eye-tracking Driving study](https://www.alphaxiv.org/abs/2408.06349)** (Khan et al., 2024):
- Participants drove a simulator in low-visibility (night + rain) conditions.
- Three n-back levels (0-back, 1-back, 2-back) were administered auditorily as a secondary task.
- Each digit presented for 1s, participants had 2.5s to respond by pressing steering wheel buttons.
- Recorded fNIRS (48-channel prefrontal cortex, NIRSIT device), eye-tracking (Pupil Core glasses), and vehicle dynamics. [n-back task](https://www.alphaxiv.org/abs/2408.06349v1?page=3)

The same n-back logic appears in the **[multi-modal cognitive-motor dataset](https://www.alphavix.org/abs/2603.22933)** (Ajra et al., 2026), but extended into a hierarchical task tree: standalone cognitive tasks (2-back, mental arithmetic) → standalone motor tasks (motor imagery, passive/active movement) → combined conditions (n-back arithmetic, full integrated cognitive-motor). 30 participants, simultaneous EEG (32-ch, 250 Hz, g.Nautilus) + fNIRS (16-ch, PFC and sensorimotor) + ECG. [Tasks](https://www.alphaxiv.org/abs/2603.22933v1?page=3)

---

### 3. The Focused Attention Meditation (FAM) Protocol

This targets **internally-generated sustained attention** — the subject must actively maintain focus on a chosen object without external stimuli driving them. Perfectly suited for studying voluntary attentional control.

**[L-FAME](https://www.alphaxiv.org/abs/2605.22893)** (Li et al., 2026) is the largest such dataset:

**Session structure (standardized sequence of 5 tasks):**
1. **Eyes-open resting** (2 min) — baseline
2. **Eyes-closed resting** (4 min) — "let your mind wander" (mind-wandering proxy)
3. **Active meditation** (8 min) — chant mantra aloud / alternate nostril breathing
4. **Eyes-closed resting** (4 min) — post-meditation rest
5. **Silent meditation** (8 min) — internal mantra repetition / breath focus (no movement, artifact-free)

**Three meditation techniques compared:**
- Breath Focus (attention on respiration sensations)
- SA-TA-NA-MA mantra (simpler mantra)
- Hare Krishna mantra (longer mantra) [EEG session](https://www.alphaxiv.org/abs/2605.22893v1?page=5)

**Longitudinal design:** 74 participants, 64-channel EEG (mBrainTrain Smarting Pro X, 10-10 system), pre- and post-6-week intervention. Each participant randomly assigned to one technique. Daily practice with gradually increasing duration (5→10→15 min/day). Daily journal tracked practice time and self-rated focus. [Longitudinal](https://www.alphaxiv.org/abs/2605.22893v1?page=4)

---

### 4. The Self-Initiated vs. Externally-Cued Attention Protocol

The most methodologically sophisticated design for teasing apart internally-generated from stimulus-driven attention.

**[Self-Initiated Attention Shifts](https://www.alphaxiv.org/abs/2605.18251)** (Zeng et al., 2026):
- Participants viewed 4 simultaneous RSVP streams at peripheral locations.
- Two conditions compared under **identical visual stimulation**:
  - **Task-constrained self-initiated (TCSI) shifts** — participant decides when to shift attention
  - **Externally instructed (EI) shifts** — an external cue tells them when to shift
- EEG segments extracted from −2.0 to −0.5 s **before** shift onset to capture preparatory activity.
- Within-subject classification of TCSI vs. EI achieved 0.962 AUC (multi-band), showing preparatory EEG carries subject-specific predictive information. Cross-subject remained at chance — emphasizing the need for **personalized models**. [Classification](https://www.alphaxiv.org/abs/2605.18251v1?page=3)

---

### 5. The Real-Time Neurofeedback Protocol

For when you want to both measure and **modulate** attention in real time.

**[NeuroPilot](https://www.alphaxiv.org/abs/2510.20958)** (Islam et al., 2025):
- **Data collection:** 20 participants watched educational videos. Novel **intra-video questionnaire** validation: questions embedded in the video itself to validate ground-truth attention vs. non-attention states.
- **Recording:** FocusCalm headband (3 dry electrodes at Fp1/Fp2/Fpz), 20 min per participant.
- **Real-time processing:** Sliding window (1750 samples, 70% overlap → prediction every ~2.1s). After 5 consecutive non-attention predictions → audio-visual alert.
- **Pilot study result:** 5 participants, 10 min video split into baseline (5 min, no feedback) vs. feedback phase (5 min, alerts on). Mean non-attention duration dropped from 37.5s to 14.97s (p = 0.007). LOSO accuracy: 88.77%. [Pilot](https://www.alphaxiv.org/abs/2510.20958v2?page=9)

---

## Summary: Protocol Design Decisions for Your Dataset Collection

| Protocol Decision | What to Choose | Rationale (from the literature) |
|---|---|---|
| **Attention induction method** | Lecture video + quiz (for naturalistic sustained attention), n-back (for graded parametric load), or meditation (for internally-generated focus) | MEMA uses videos; L-FAME uses meditation; driving studies use n-back |
| **Number of states** | 3 (neutral, relaxing, concentrating) for categorical; 4+ (baseline + n levels) for graded | 3-state gives richer labels; binary is simplest but loses nuance |
| **State duration** | 1-8 min per block | Short (1 min) for neutral; longer (5-8 min) for focused states to allow neural entrainment |
| **Validation mechanism** | Embedded comprehension questions (intra-video) + self-assessment + behavioral performance (reaction time, accuracy) | NeuroPilot shows intra-video Qs minimize recall bias; MEMA uses post-trial self-assessment + quiz |
| **Modalities** | EEG (high temporal resolution, gold standard) + optionally fNIRS (spatial resolution, hemodynamic) + eye-tracking (attention proxy) | 32-64 ch EEG is standard; fNIRS adds PFC coverage; eye-tracking gives behavioral correlate |
| **Subjects** | 20-74 is the range in published work | More subjects = better cross-subject generalization, but even 20 works (MEMA: 20, NeuroPilot: 20) |
| **Trials per subject** | 12-72 events per condition | More trials = better SNR; MEMA: 12 trials/subject; n-back tasks: 72 events |
| **Longitudinal vs. cross-sectional** | Cross-sectional for quick first dataset; longitudinal (pre/post 6-week intervention) for studying training effects | L-FAME shows cross-session generalization drops from ~97% intra to ~63-67% zero-shot — temporal drift is real |
| **Labeling strategy** | Self-report + behavioral validation + EEG feature consistency checks (alpha/beta ratio) | MEMA validated via topographic maps (alpha power ↑ as attention ↓, beta power ↑ as attention ↑) |
| **EEG preprocessing** | 0.5-30/50 Hz bandpass, ICA for artifact removal, 50 Hz notch, high-amplitude rejection (>150 μV) | Common across all datasets |
| **Key pitfalls to avoid** | Cross-subject variance dominates — expect 60-65% cross-subject vs. 85-97% within-subject performance | Universal finding across MEMA, L-FAME, and attention-shift studies |

---

## Where the Field Is Moving (For Your Dataset Planning)

1. **Multi-label and multi-modal is the standard now.** MEMA collects emotion labels alongside attention; the cognitive-motor dataset collects EEG+fNIRS+ECG+behavioral simultaneously. Single-modality, single-label datasets are no longer competitive.

2. **Cross-session temporal drift is a first-class problem.** L-FAME's Task 3 shows zero-shot performance dropping from ~97% intra-session to ~63% when tested 6 weeks later — and that's on the *same subjects*. If you plan longitudinal collection, you need few-shot adaptation built into your protocol.

3. **Consumer-grade hardware is viable.** The NeuroPilot study achieved 88.77% LOSO accuracy with a 3-electrode dry EEG headband. You do not need a research-grade 256-channel system to produce publishable attention datasets — but you do need rigorous ground-truth validation.

4. **Intra-video/online validation beats post-hoc labeling.** The NeuroPilot intra-video questionnaire approach produces more reliable ground truth than post-task recall. For any naturalistic attention dataset, embed validation probes into the stimulus itself.

5. **High-frequency bands (gamma, high beta) carry strong discriminative signal** for self-initiated attention — but be cautious of muscular artifact. The attention-shift study found gamma AUC of 0.964, comparable to the multi-band model, but caveats it as possibly contaminated by residual EMG. [Results](https://www.alphaxiv.org/abs/2605.18251v1?page=3)

All of these datasets are open-access or publicly available, linked above. Would you like me to go deeper into any specific protocol family — the meditation framework, the n-back parametric design, or the real-time neurofeedback loop?