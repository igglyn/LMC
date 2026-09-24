# Latent thinking-window compression experiment

## Objective

Train the full weights of `LiquidAI/LFM2.5-8B-A1B` to preserve requested final
outputs while shifting selected reasoning steps from a serialized path into a
latent path.  This is not adapter training and is not textual chain-of-thought
summarization.

## Window and routing contract

Each example contains a bounded, causal thinking window and its requested final
response.  A co-trained compressor produces a bitfield `b` over the thinking
steps.  The bit is a training aid rather than a required final-model component.

* `b[t] = 1`: the step uses the normal non-latent/serializable path.
* `b[t] = 0`: the step uses the latent path.  Its causal K/V contribution stays
  available until the current thinking window closes; it is not required to
  take a decoded-text then re-encoded-text round trip.

The compressor is limited to information causally available in its current
thinking window.  It cannot consult future windows, the final answer, or a
post-hoc verifier when making an inference-time decision.  K/V state created
for latent steps is cleared at the window boundary.

## Loss and evaluation

The primary constraint is final-output consistency with the teacher target or a
task verifier.  Compression pressure is learned jointly, rather than making a
zero bit destructive.  A fully latent window is a valid outcome when it
preserves the required output.

Every run must report separately:

* latent and non-latent thinking-token counts;
* total internal generation/forward steps;
* K/V cache length and memory use;
* final-output consistency or verifier score; and
* the quality/compression Pareto frontier used for checkpoint selection.

The first positional baseline preserves original positions.  Compacted
positions are an explicit experimental variant, not an assumed requirement.

## Teacher boundary and future concurrency

An external agent harness owns tools, sessions, and raw trace collection.  LMC
accepts raw trace records or normalized thought/response pairs.  LMC owns local
teacher requests through a direct `transformers` runner rather than an external
model-server interface.  It loads one checkpoint and batches up to
`teacher.max_concurrent_requests` logical requests at a time; it never assumes
one GPU replica per worker.  The runner requires the optional `runtime` extra
and only loads local model artifacts.

## Reproducibility

Finalized datasets use stable example IDs and immutable manifests.  Training
checkpoints must eventually include model, optimizer, scheduler, scaler,
sampler, RNG, and complete run configuration state.  Raw reasoning is not sent
to W&B by default.
