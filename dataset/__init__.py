"""Multi-device dataset build + dataloader pipeline.

Turns one raw participant folder under
C:\\emotion-stress-attention-task-ear-wristband-device-data\\data-collection-sessions
into a canonical, host-UTC-synced set of per-stream Parquet files plus a
sync/QC report, and provides an event-windowed dataloader on top of that.

See docs/Dataset_Sync_Design.md for the full design write-up (sampling rates,
sync-anchor formulas per device, restart/opt-out resolution rules).
"""
