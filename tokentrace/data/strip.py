"""Reference-stripping at the :class:`~tokentrace.core.types.Inference` level.

A production trace carries no ``ground_truth`` and no ``Chunk.gold`` annotation;
these helpers put an offline trace into that shape. They live in ``data/`` —
below both :mod:`tokentrace.eval.benchmark` and :mod:`tokentrace.eval.noref` in
the import graph — because both need them: ``eval/noref`` wraps
:func:`strip_inference` into the labeled-aware ``strip_reference`` used by the
§4.5 experiments, and ``eval/benchmark`` uses it to add reference-stripped
passes to the calibration records so genuine ``|noref`` maps exist. Importing
the helper from ``eval/noref`` instead would create the import cycle
``benchmark -> noref -> benchmark``.

The semantics (why gold flags become ``False`` rather than ``None``, why
``meta["_sim"]`` is deliberately kept, what stripping does and does not
guarantee) are documented once, on ``eval/noref.strip_reference`` — the
user-facing wrapper. This module is only the mechanism.
"""

from __future__ import annotations

import copy

from tokentrace.core.types import Inference


def strip_inference_in_place(inference: Inference, *, drop_gold_flags: bool = True,
                             drop_ground_truth: bool = True) -> Inference:
    """Remove the reference metadata from ``inference`` itself, and return it.

    The single definition of what "stripped" means, shared by
    :func:`strip_inference` and ``eval/noref.strip_reference`` so the two cannot
    drift. Callers that need the original preserved must copy first — every
    public entry point in this codebase does.
    """
    if drop_ground_truth:
        inference.ground_truth = None
    if drop_gold_flags:
        for chunk in inference.retrieved_context or ():
            chunk.gold = False
    return inference


def strip_inference(inference: Inference, *, drop_gold_flags: bool = True,
                    drop_ground_truth: bool = True) -> Inference:
    """A deep copy of ``inference`` in the shape a production trace arrives in.

    A copy, not an in-place edit: the reference-bearing original is typically the
    control group its stripped twin is compared against.
    """
    return strip_inference_in_place(
        copy.deepcopy(inference),
        drop_gold_flags=drop_gold_flags, drop_ground_truth=drop_ground_truth,
    )
