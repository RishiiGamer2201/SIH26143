"""oilwatch: SIH26143 oil-spill attribution pipeline.

Stages wrap the frozen experiment scripts (never edited; see PROJECT_CONTEXT.md "Hard links") and write
only into run-scoped folders runs/<incident>/<run_id>/<stage>/ with a manifest.json per stage.
"""
__version__ = "0.1.0"
