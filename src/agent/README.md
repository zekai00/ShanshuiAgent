# Legacy Agent Package

`src/agent/` contains the earlier command-line multi-agent prototype.

The current product path is `src/web_agent/`, which implements the LangGraph
web-agent workflow used by `scripts/run_web_app.py` and `/api/agent/stream`.

Keep this package for reference, experiments, and legacy CLI runs, but new
research, verification, image-generation, health-check, and smoke-test work
should target `src/web_agent/` first.

Useful distinction:

- `src/web_agent/`: current evidence-grounded LangGraph multi-role workflow.
- `src/agent/`: legacy supervisor/researcher/artist/chatter command-line graph.
