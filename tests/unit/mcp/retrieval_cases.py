"""Ground-truth cases for the MCP retrieval eval suite.

Each case is a natural-language question an AI client might ask, paired with the
repo-relative path fragment of the source that should answer it. The eval
(``test_retrieval_eval.py``) runs every question through
``search_knowledge_sources`` against the real repository knowledge base and
checks that the expected source is among the top results.

Keep the ``expected`` fragments specific enough to identify one file but short
enough to survive small renames. When you add or move a doc page, add or update
the relevant case here so ranking regressions stay visible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected: str  # path fragment that must appear in a hit's ``path``


CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "How do I set the simulation wavelength grid?",
        "focal_plane/general",
    ),
    RetrievalCase(
        "What does the oversampling factor do to the focal plane?",
        "focal_plane/resulting_focal_plane",
    ),
    RetrievalCase(
        "How are foreground contributions like zodiacal light added?",
        "focal_plane/foregrounds",
    ),
    RetrievalCase(
        "How is pointing jitter applied to the sub-exposures?",
        "sub-exposures/pointing_jitter",
    ),
    RetrievalCase(
        "How is the transit light curve imprinted on the star?",
        "sub-exposures/astronomical_signal",
    ),
    RetrievalCase(
        "How are non-destructive reads accumulated into the ramp?",
        "ndrs/accumulate",
    ),
    RetrievalCase(
        "How is read noise added to the detector reads?",
        "ndrs/readnoise",
    ),
    RetrievalCase(
        "How does ExoSim model cosmic ray hits?",
        "ndrs/cosmic_rays",
    ),
    RetrievalCase(
        "How is the analogue to digital conversion and gain handled?",
        "ndrs/analogtodigtital",
    ),
    RetrievalCase(
        "How does the radiometric model compute photon noise?",
        "radiometric_model/photon_noise",
    ),
    RetrievalCase(
        "How are the extraction apertures estimated per wavelength bin?",
        "radiometric_model/estimate_apertures",
    ),
    RetrievalCase(
        "How do I run the radiometric model over a target list?",
        "radiometric_model/operating_modes",
    ),
    RetrievalCase(
        "How do I write a custom Task in ExoSim?",
        "contributing/custom_tasks",
    ),
    RetrievalCase(
        "What is RunConfig and how do I set the number of threads?",
        "runConfig",
    ),
    RetrievalCase(
        "How do I load an HDF5 output file and read a signal table back?",
        "FAQs",
    ),
    RetrievalCase(
        "How do I connect an MCP client to the ExoSim server?",
        "user/mcp",
    ),
    RetrievalCase(
        "How do I add the ExoSim MCP server to VS Code or Claude Desktop?",
        "user/mcp",
    ),
    RetrievalCase(
        "How do I change the telescope collecting area or the target star?",
        "user/common_edits",
    ),
)
