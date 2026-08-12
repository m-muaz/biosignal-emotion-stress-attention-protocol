<#
.SYNOPSIS
    Runs the full data-collection protocol for one participant: the main
    session (consent/questionnaire -> emotion -> stress -> attention/focus)
    followed by SART, with a short countdown in between.

.DESCRIPTION
    SART is deliberately kept as a separate standalone process from
    `app.main` (see README's "Attention task (SART)" section, and
    [[project_attention_task_sart]] -- per PI request, not wired into the
    main session flow/time budget). This script just chains the two
    invocations you'd otherwise run by hand:

        python -m app.main --participant-id <id> --devices-mode real --skip-familiarization
        python -m app.run_task --task sart --participant-id <id> --devices-mode real --skip-practice

    Both `--skip-familiarization` (main session) and `--skip-practice`
    (SART) assume the participant has already been walked through a demo
    of every task beforehand, so neither script needs to show its own
    in-app practice/familiarization round.

    Run this from an already-activated `data_collection` conda prompt,
    exactly like you'd run the two commands manually -- this script does
    NOT activate the environment for you.

    If the main session is aborted (Escape) or crashes, SART is NOT
    started -- see app/main.py's/app/run_task.py's exit codes
    (0 = completed, 2 = operator aborted, 1 = crashed).

.PARAMETER ParticipantId
    Required. Same participant id used for both the main session and SART
    (each still gets its own separate session folder under ./sessions/).

.PARAMETER DevicesMode
    "real" (default) or "mock". Passed through to both commands.

.PARAMETER SkipDeviceSync
    Pass this if hardware is already synced/verified via external scripts
    before this script starts (same meaning as the flag on both
    app.main/app.run_task). Applied to BOTH commands.

.PARAMETER CountdownSeconds
    Pause between the main session ending and SART starting. Default 5.

.PARAMETER Fullscreen
    Run SART's psychopy window fullscreen (passes --fullscreen to
    app.run_task). Only affects SART -- the main session's tasks are all
    browser-based (their own fullscreen handling, if any, lives in the web
    UI) and app.main has no --fullscreen flag to pass. On by default.

.EXAMPLE
    ./run_full_session.ps1 -ParticipantId shruti

.EXAMPLE
    ./run_full_session.ps1 -ParticipantId shruti -DevicesMode mock -CountdownSeconds 10

.EXAMPLE
    ./run_full_session.ps1 -ParticipantId shruti -Fullscreen:$false
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$ParticipantId,

    [ValidateSet("real", "mock")]
    [string]$DevicesMode = "real",

    [switch]$SkipDeviceSync,

    [int]$CountdownSeconds = 5,

    [bool]$Fullscreen = $true
)

function Write-Section($text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

Write-Section "MAIN SESSION -- consent/questionnaire -> emotion -> stress -> attention/focus (participant: $ParticipantId)"

$mainArgs = @(
    "-m", "app.main",
    "--participant-id", $ParticipantId,
    "--devices-mode", $DevicesMode,
    "--skip-familiarization"
)
if ($SkipDeviceSync) { $mainArgs += "--skip-device-sync" }

python @mainArgs
$mainExit = $LASTEXITCODE

if ($mainExit -eq 2) {
    Write-Host "`nMain session was ABORTED by the operator (Escape) -- not starting SART." -ForegroundColor Yellow
    exit 2
} elseif ($mainExit -ne 0) {
    Write-Host "`nMain session CRASHED (exit code $mainExit) -- not starting SART. Check the traceback above and this participant's events.jsonl." -ForegroundColor Red
    exit $mainExit
}

Write-Host "`nMain session complete for '$ParticipantId'." -ForegroundColor Green

for ($i = $CountdownSeconds; $i -gt 0; $i--) {
    Write-Host "`rStarting SART in $i...  " -NoNewline -ForegroundColor Yellow
    Start-Sleep -Seconds 1
}
Write-Host "`rStarting SART now.        "

Write-Section "SART -- Sustained Attention to Response Task (participant: $ParticipantId)"

$sartArgs = @(
    "-m", "app.run_task",
    "--task", "sart",
    "--participant-id", $ParticipantId,
    "--devices-mode", $DevicesMode,
    "--skip-practice"
)
if ($SkipDeviceSync) { $sartArgs += "--skip-device-sync" }
if ($Fullscreen) { $sartArgs += "--fullscreen" }

python @sartArgs
$sartExit = $LASTEXITCODE

if ($sartExit -eq 0) {
    Write-Host "`nFull protocol (main session + SART) complete for '$ParticipantId'." -ForegroundColor Green
} elseif ($sartExit -eq 2) {
    Write-Host "`nSART was ABORTED by the operator (Escape)." -ForegroundColor Yellow
} else {
    Write-Host "`nSART CRASHED (exit code $sartExit). Check the traceback above and this participant's events.jsonl." -ForegroundColor Red
}

exit $sartExit
