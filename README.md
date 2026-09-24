# LMC

LMC is a spec-first experimental framework for training a full-weight language
model with latent thinking-window compression.  A co-trained compressor emits a
per-step bitfield: normal steps may be serialized, while latent steps retain
their causal K/V contribution inside the active thinking window without a text
round trip.

This repository deliberately does **not** implement an agent harness or a
general model server.  It consumes teacher traces supplied by an external
harness.  A direct, in-repository model runtime will later own teacher requests
and the training forward pass; no external model-server integration is part of
this milestone.

## Status

The first milestone establishes reproducible configuration, trace schemas, and
normalization and a direct local teacher runner.  The model patch and
full-weight training backend are specified in [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)
but remain later milestones.

## Quick start

```bash
python -m pip install -e '.[dev]'
cp configs/experiment.example.json experiment.json
python -m lmc normalize --input examples/traces.jsonl --output data/pairs.jsonl
python -m lmc validate --input data/pairs.jsonl
```

Install the runtime extra to run the downloaded local model:

```bash
python -m pip install -e '.[runtime]'
python -m lmc generate --config experiment.json --input requests.jsonl --output data/teacher.jsonl
```

Each `requests.jsonl` record contains a stable `id` plus exactly one of a
`prompt` string or a chat `messages` list.  The runner batches up to
`teacher.max_concurrent_requests` logical requests against one loaded model; it
does not load one model replica per worker.

## Data boundary

Raw traces are source records.  Normalized thought/response pairs are derived
artifacts and are the input to later routing and training stages.  Keep raw
reasoning data in controlled storage and avoid uploading it to experiment
tracking by default.
