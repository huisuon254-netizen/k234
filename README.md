# k234 - AI Agent Task Server

## Structure
```
k234/
├── context/          → Stored knowledge (shape DNA, blueprints, evolved insights)
├── tasks/            → 100+ task definitions (JSON)
├── results/          → Agent outputs for each task
├── analyses/         → Gemini evaluations of each agent's work
├── workflows/        → Task chain / workflow definitions
├── wave_output/      → Per-wave aggregated reports
├── reports/          → Master reports
├── agent_logs/       → Per-agent execution logs
├── agent_orchestrator.py  → Main orchestrator (105 tasks, 9 waves)
├── push_to_github.sh → Push to GitHub server
└── master_report.json     → Latest execution summary
```

## How This Repo Serves As A Server

This GitHub repo acts as a **storage + workflow server**:

1. **Storage**: Large files (JSON results, analysis data, context) are stored via git
2. **Workflows**: Scripts in the repo define task chains (waves) that agents execute
3. **Small tasks**: Defined as JSON in `tasks/`, processed by Groq/LLM agents
4. **Analysis**: Gemini reviews every agent's output in `analyses/`
5. **Knowledge growth**: `context/` evolves as tasks complete → feeds back into next wave

## Usage

### Run all 105 agent tasks (9 waves):
```bash
python3 agent_orchestrator.py
```

### Push to GitHub (server sync):
```bash
GITHUB_TOKEN=your_token_here bash push_to_github.sh
```

### Results from 2026-05-19 run:
- 105 tasks across 9 waves
- 80/105 completed (25 Groq rate-limited on retry exhaustion)
- 8 agents: ShapeDNA, GeometryCritic, DesignEnhancer, ProportionAnalyzer, GameDesigner, MaterialScientist, EvolutionMutator, QualityScorer, TaskPlanner
- Knowledge from shape_dna.json + evolved_knowledge.json fed into task generation

