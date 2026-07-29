# Data Collection Protocol

A single-visit protocol comprising three tasks, conducted in the
following order: **Attention → Emotion → Stress**. Estimated total
duration: approximately 3 hours.

  -----------------------------------------------------------------------
  **Phase**                                          **Duration**
  -------------------------------------------------- --------------------
  Preparation (informed consent, questionnaire,      25 min
  electrode setup)                                   

  Task 1: Attention                                  \~55 min

  Rest                                               15 min

  Task 2: Emotion                                    \~30 min

  Rest                                               30--45 min

  Task 3: Stress                                     \~10 min

  Debriefing                                         5 min
  -----------------------------------------------------------------------

## Task 1: Attention (adapted from Liu et al., 2024, MEMA)

### Materials to Prepare

**Neutral video (1 minute)**: Produce a 1-minute video file consisting
of a single solid-colored or blank screen (e.g., mid-gray, RGB
128/128/128), with no visual or auditory content, exported as .mp4. This
can be created with any basic video editing software (e.g., a static
image extended to 1 minute using Adobe Premiere, DaVinci Resolve, or a
free tool such as Shotcut).

**Relaxing video (5 minutes)**: Produce a 5-minute video combining
nature/landscape footage with soothing instrumental music, following the
content description in the original MEMA study (\"a five-minute soothing
video of beautiful scenery, accompanied by relaxing music\"). Concrete
steps:

> 1\. Source royalty-free nature/landscape footage (e.g., ocean waves,
> forests, mountains) from stock video libraries such as Pexels Video or
> Pixabay Video (both offer free, license-free downloads).
>
> 2\. Source royalty-free ambient/instrumental background music from the
> same platforms or from YouTube Audio Library (free, license-free).
>
> 3\. Combine footage and music into a single 5-minute video using
> standard video editing software.

**Concentrating video (5 minutes) + comprehension question**: The
original MEMA study used a 5-minute segment from Andrew Ng\'s Machine
Learning course on Coursera (Coursera Machine Learning Specialization,
[https://www.coursera.org/specializations/machine-learning-introduction]{.underline}).
Concrete steps:

> 1\. Select one lecture video (approximately 5 minutes) from Course 1
> (\"Supervised Machine Learning: Regression and Classification\") of
> this specialization, covering a single, well-defined concept (e.g.,
> the definition of linear regression).
>
> 2\. Prepare one multiple-choice comprehension question based directly
> on the core concept explained in the selected segment (4 options, 1
> correct answer), to be presented immediately after the video.

**Physical setup**: Same viewing setup as used for the other two tasks
(participant seated in front of a monitor with headphones or speakers
for audio playback).

### Label Acquisition

The label is determined by task/video type assignment (i.e., which of
the three conditions --- neutral, relaxing, or concentrating --- the
trial belongs to), fixed at the design stage.

### Participant Procedure (per trial)

  ------------------------------------------------------------------------
  **Step**           **Duration**       **Action**
  ------------------ ------------------ ----------------------------------
  Cue                5 s                An auditory cue signals that the
                                        trial is about to begin

  Video viewing      Neutral: 1 min /   Neutral: watch the screen
                     Relaxing: 5 min /  normally, without deliberately
                     Concentrating: 5   relaxing or concentrating;
                     min                Relaxing: try to relax as much as
                                        possible; Concentrating: watch
                                        attentively and actively think
                                        about the content

  Comprehension      \~1 min            Answer the multiple-choice
  question                              question related to the video just
  (Concentrating                        viewed
  condition only)                       

  Rest               15 s               Quiet rest, no task
  ------------------------------------------------------------------------

Each round consists of one trial of each of the three conditions
(neutral, relaxing, concentrating), with the order of the three
conditions randomized within each round and across participants. Four
rounds are completed in total, yielding 12 trials per participant.

## Task 2: Emotion (adapted from Chen et al., 2023, FACED)

### Materials to Prepare

**Emotion category structure** (based on the complete FACED taxonomy of
nine discrete emotion categories --- four positive, four negative, one
neutral):

- **Positive-valence**: amusement, inspiration, joy, tenderness (all
  four positive sub-emotions from the original FACED taxonomy)

- **Negative-valence**: anger, disgust, fear, sadness (all four negative
  sub-emotions)

- **Neutral**: emotionally neutral content

**Selected film clips** (one clip per sub-emotion at minimum, four clips
per valence category, four clips for neutral --- matching FACED\'s
neutral clip count):

  -------------------------------------------------------------------------------------------
  **Category**       **Sub-emotion**   **Clip**               **Source**
  ------------------ ----------------- ---------------------- -------------------------------
  Positive-valence   Amusement         *When Harry Met Sally* Gross & Levenson (1995)
                                       (1989), restaurant     validated stimulus set
                                       scene                  

  Positive-valence   Amusement         Robin Williams         Gross & Levenson (1995)
                     (alternate)       stand-up comedy        validated stimulus set
                                       excerpt                

  Positive-valence   Tenderness        Tenderness-category    FilmStim (Schaefer et al.,
                                       clip                   2010) --- request from database
                                                              (see below)

  Positive-valence   Inspiration       Inspiration-category   Hu et al. (2017), *Frontiers in
                                       clip                   Human Neuroscience*, \"database
                                                              of positive emotional videos\"
                                                              --- request from corresponding
                                                              author (see below)

  Positive-valence   Joy               Joy-category clip      Hu et al. (2017) positive
                                                              emotional video database ---
                                                              request from corresponding
                                                              author

  Negative-valence   Sadness           *The Champ* (1979),    Gross & Levenson (1995)
                                       closing death scene    validated stimulus set

  Negative-valence   Anger             *My Bodyguard* (1980)  Gross & Levenson (1995)
                                       or *Cry Freedom*       validated stimulus set
                                       (1987),                
                                       injustice-themed scene 

  Negative-valence   Disgust           Disgust-category clip  FilmStim (Schaefer et al.,
                                                              2010) --- request from database

  Negative-valence   Fear              Fear-category clip     FilmStim (Schaefer et al.,
                                                              2010) --- request from database

  Neutral            ---               Four clips of          Standard neutral stimuli used
                                       weather-report or      in both Gross & Levenson (1995)
                                       nature-documentary     and FilmStim (Schaefer et al.,
                                       footage                2010); publicly available
                                                              nature/weather footage may also
                                                              be used as neutral stimuli
  -------------------------------------------------------------------------------------------

Each clip is edited to 1--2 minutes in duration.

**How to obtain the clips that require a request**:

> 1\. **FilmStim** (tenderness, disgust, fear categories): submit a data
> request through the official FilmStim website at
> nemo.psp.ucl.ac.be/FilmStim to obtain the validated clip files, exact
> edit points, and accompanying normative rating data.
>
> 2\. **Hu et al. (2017) positive emotional video database**
> (inspiration, joy categories): contact the corresponding author of the
> paper (Dan Zhang, Department of Psychology, Tsinghua University; the
> same laboratory that later produced the FACED dataset) to request the
> stimulus materials, citing the published paper (Hu et al., 2017,
> *Frontiers in Human Neuroscience*, DOI: 10.3389/fnhum.2017.00026).

**Physical setup**: participant seated approximately 60 cm from the
display monitor; audio played through stereo speakers (following the
FACED original setup).

### Label Acquisition

The label is determined by block assignment (i.e., which category of
video the trial belongs to) --- positive-valence / negative-valence /
neutral. This label is fixed at the design stage (based on which clip is
shown) and does not depend on participant self-report.

### Participant Procedure (per trial)

  -----------------------------------------------------------------------
  **Step**           **Duration**   **Action**
  ------------------ -------------- -------------------------------------
  Fixation           5 s            Fixate on a cross displayed at the
                                    center of the screen

  Video viewing      1--2 min       Watch the clip quietly, without
                                    speaking or moving

  Rest               30 s           Blank screen, quiet rest
  -----------------------------------------------------------------------

**Block structure**: clips of the same valence category are presented
consecutively as a block, yielding three blocks (positive-valence /
negative-valence / neutral); the order of the three blocks is randomized
across participants, and the order of clips within each block is also
randomized. Between two consecutive blocks, the participant completes 20
serial-subtraction arithmetic problems (each problem consists of a
4-digit number minus a 1--2-digit number; if a problem is not answered
within 4 seconds, it is automatically skipped and the next problem is
presented). One practice trial precedes the formal session to
familiarize the participant with the procedure.

## Task 3: Stress (adapted from Zyma et al., 2019, EEGMAT)

### Materials to Prepare

**Arithmetic problem**: one pair of numbers --- a 4-digit minuend and a
1--2-digit subtrahend (e.g., \"4753\" and \"17\") --- to be read aloud
to the participant at the start of the task. The numbers may be
generated at random; only one such pair is needed per participant, since
the task consists of a single continuous 4-minute subtraction session
(not multiple repeated trials).

**Physical setup**: following the original EEGMAT protocol, the session
should take place in a dark, quiet (sound-attenuated) room, with the
participant seated comfortably in a reclining chair.

**Timer**: to track the three consecutive phases (3 min adaptation, 3
min resting baseline, 4 min arithmetic task).

**Instructions to give the participant before starting** (based on the
original protocol): explain that during the resting baseline they should
sit quietly with eyes closed; explain that during the arithmetic task
they should count mentally, without speaking aloud or using finger
movements, working as quickly and accurately as possible at their own
pace, and that they will report their final result once the task period
ends.

### Label Acquisition

- **Primary label (binary classification)**: determined by which phase
  the EEG segment was recorded in --- the resting baseline segment is
  labeled *no-stress/rest*, and the arithmetic task segment is labeled
  *stress/task*. This is the primary label used for stress
  classification and is fixed by the recording condition itself, not by
  any rating.

- **Secondary behavioral measures (for later grouping/analysis, not a
  regression target)**: at the end of the task, the participant orally
  reports the final numerical result they reached. The experimenter then
  computes the number of completed operations --- (initial 4-digit
  number − final reported result) ÷ subtrahend --- and the accuracy of
  the result (considered valid if it does not differ by more than 20%
  from the mathematically correct value). These continuous measures are
  not themselves the classification label; they may be used afterward to
  divide participants into two performance-based groups (e.g.,
  higher-performing vs. lower-performing), following the same logic as
  the original EEGMAT study\'s grouping procedure.

### Participant Procedure

  -------------------------------------------------------------------------
  **Phase**      **Duration**   **Action**
  -------------- -------------- -------------------------------------------
  Adaptation     3 min          Sit quietly in the reclining chair to adapt
                                to the recording environment

  Resting        3 min          Sit with eyes closed, without performing
  baseline                      any task

  Serial         4 min,         Upon hearing the two numbers (e.g.,
  subtraction    continuous     \"4753\" and \"17\"), perform mental
  task                          subtraction silently (without speaking or
                                using finger movements), repeatedly
                                subtracting the subtrahend from the running
                                result (4753 → 4736 → 4719 → ...) as
                                quickly and accurately as possible, at a
                                self-determined pace, until the task period
                                ends

  Reporting      ---            Orally report the final result reached
  -------------------------------------------------------------------------
