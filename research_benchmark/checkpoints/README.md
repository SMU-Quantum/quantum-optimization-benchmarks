# Batch Checkpoints

This directory stores checkpoint JSON files for resumable benchmark sweeps.

Each file records the instances already completed for a given method/problem combination so a long run can be resumed without recomputing finished work.

The packaged hardware CLI defaults to this directory for checkpointing. The simulator-only workflow uses `research_benchmark/research_benchmark/simulator_checkpoints/` instead so simulator resumptions stay separate from hardware resumptions.
