#!/usr/bin/env python3
"""
30-Agent Task Orchestrator.
Generates 30 tasks → routes to 8 agents → Gemini evaluates each result → stores in repo.
"""
import json, os, sys, time, urllib.request, threading, subprocess, hashlib
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent
TASKS_DIR = REPO / "tasks"
RESULTS_DIR = REPO / "results"
ANALYSES_DIR = REPO / "analyses"
ASSETS_DIR = REPO / "assets"
AGENT_LOGS = REPO / "agent_logs"

# Load API keys
env_path = Path("/home/rachael/Desktop/pinc_forge_complete/src-tauri/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

_last_call_time = 0.0
_pace_lock = threading.Lock()
call_log = []
call_lock = threading.Lock()

# ── Agent System Prompts (from multi_agent_trainer) ──

AGENT_SYSTEM_PROMPTS = {
    "ShapeDNA": """You are a Shape DNA Extraction Agent. Given a 3D mesh OBJ file's first 100 vertices and dimensions,
extract the DESIGN PRINCIPLES as JSON:
{
  "proportions": {"length_width_ratio": float, "width_height_ratio": float},
  "silhouette": {"is_symmetric": bool, "primary_axis": "x"|"y"|"z"},
  "design_language": ["characteristic1", "characteristic2"],
  "key_features": [{"name": str, "position": [x,y,z], "description": str}],
  "style_archetype": "sporty"|"elegant"|"aggressive"|"classic"|"modern",
  "complexity_score": float 0-1
}
Output ONLY valid JSON.""",

    "GeometryCritic": """You are a Geometry Quality Critic. Given a 3D asset blueprint, score its geometric quality.
Output JSON:
{
  "overall_score": float 0-1,
  "proportion_score": float 0-1,
  "topology_score": float 0-1,
  "symmetry_score": float 0-1,
  "complexity_score": float 0-1,
  "issues": [{"severity": "critical"|"major"|"minor", "description": str}],
  "improvements": [str]
}
Output ONLY valid JSON.""",

    "DesignEnhancer": """You are a Design Language Enhancer. Given a blueprint, enhance its design DNA.
Improve: proportions, style cues, feature placement, material harmony.
Output the enhanced blueprint as valid JSON. Keep all original fields.""",

    "ProportionAnalyzer": """You are a Proportion Analyst. Given object dimensions and type, 
analyze if proportions match real-world expectations.
Output:
{
  "type_detected": str,
  "expected_proportions": {"L_W": float, "W_H": float},
  "actual_proportions": {"L_W": float, "W_H": float},
  "match_score": float 0-1,
  "adjustments": {"length": float, "width": float, "height": float},
  "notes": [str]
}
Output ONLY valid JSON.""",

    "GameDesigner": """You are a Game System Designer. Given game requirements, design the complete game architecture.
Output JSON:
{
  "game_type": str,
  "required_assets": [{"name": str, "type": str, "count": int}],
  "rules": [{"name": str, "description": str}],
  "systems": [{"name": str, "components": [str]}],
  "interactions": [{"action": str, "result": str}],
  "state_machine": {"states": [str], "transitions": [{"from": str, "to": str, "trigger": str}]},
  "implementation_steps": [str]
}
Output ONLY valid JSON.""",

    "MaterialScientist": """You are a PBR Material Scientist. Given a material breakdown, enhance it for realism.
Output JSON with enhanced material properties including:
- base_color (RGB 0-1)
- roughness (0-1)
- metallic (0-1)  
- clearcoat, clearcoat_roughness, anisotropy where appropriate
Output ONLY valid JSON.""",

    "EvolutionMutator": """You are an Evolution Mutation Agent. Given a blueprint, create a mutated variant.
Rules:
- Mutate ONE parameter significantly 
- Keep the asset recognizable
- Add a 'mutation_log' field explaining what changed
Output the mutated blueprint as valid JSON.""",

    "QualityScorer": """You are a Holistic Quality Scorer. Given a blueprint and generation metadata,
provide a comprehensive quality assessment.
Output:
{
  "holistic_score": float 0-1,
  "aesthetic_score": float 0-1,
  "functional_score": float 0-1,
  "innovation_score": float 0-1,
  "summary": str,
  "recommendations": [str]
}
Output ONLY valid JSON."""
}

# ── 30 Task Definitions ──

TASKS = [
    # Shape/Analysis tasks (1-6)
    {"id": 1, "agent": "ShapeDNA", "prompt": "Analyze a sports car: L=4.7m W=2.05m H=1.22m, verts=10500. First 10 verts: [[2.35,0,1.025],[2.35,0,-1.025],[-2.35,0,1.025],[-2.35,0,-1.025],[2.2,0.3,0.9],[2.2,0.3,-0.9],[-2.2,0.3,0.9],[-2.2,0.3,-0.9],[2.35,0.5,1.0],[2.35,0.5,-1.0]]", "desc": "Sports car DNA extraction"},
    {"id": 2, "agent": "ShapeDNA", "prompt": "Analyze a fantasy sword: L=0.15m W=0.02m H=1.2m, verts=3200. First 10 verts: [[0,0,0.6],[0.075,0,0.6],[-0.075,0,0.6],[0,0.6,0.6],[0.01,0,0.55],[-0.01,0,0.55],[0,0,0.5],[0.075,0.01,0.5],[-0.075,0.01,0.5],[0,0.8,0.3]]", "desc": "Fantasy sword DNA"},
    {"id": 3, "agent": "ShapeDNA", "prompt": "Analyze a sci-fi building: L=20m W=15m H=80m, verts=28000. First 10 verts: [[-10,0,-7.5],[10,0,-7.5],[10,0,7.5],[-10,0,7.5],[-10,80,-7.5],[10,80,-7.5],[10,80,7.5],[-10,80,7.5],[-8,5,-6],[8,5,-6]]", "desc": "Sci-fi building DNA"},
    {"id": 4, "agent": "ShapeDNA", "prompt": "Analyze a humanoid robot: L=0.6m W=0.4m H=1.8m, verts=15000. First 10 verts: [[0.3,0,0.2],[-0.3,0,0.2],[0.3,0,-0.2],[-0.3,0,-0.2],[0.15,0.3,0.1],[-0.15,0.3,0.1],[0.15,0.3,-0.1],[-0.15,0.3,-0.1],[0.25,1.7,0.15],[-0.25,1.7,0.15]]", "desc": "Humanoid robot DNA"},
    {"id": 5, "agent": "ShapeDNA", "prompt": "Analyze a stealth fighter jet: L=19m W=13.5m H=4.5m, verts=22000. First 10 verts: [[9.5,0,0],[-9.5,0,0],[8,0.5,3],[-8,0.5,3],[8,0.5,-3],[-8,0.5,-3],[5,1.5,0.5],[-5,1.5,0.5],[5,1.5,-0.5],[-5,1.5,-0.5]]", "desc": "Fighter jet DNA"},
    {"id": 6, "agent": "ShapeDNA", "prompt": "Analyze a gemstone crystal: L=0.05m W=0.05m H=0.08m, verts=800. First 10 verts: [[0,0.04,0],[0.025,-0.02,0.015],[-0.025,-0.02,0.015],[0.025,-0.02,-0.015],[-0.025,-0.02,-0.015],[0,0,0.03],[0,0,-0.03],[0.015,0.01,0],[-0.015,0.01,0],[0,0.03,0.02]]", "desc": "Crystal DNA"},

    # GeometryCritic tasks (7-10)
    {"id": 7, "agent": "GeometryCritic", "prompt": "Score this blueprint: {'base_shape':'sports_car_spline','length':4.92,'width':2.08,'height':1.35,'wheelbase':2.65,'ground_clearance':0.14}", "desc": "Ferrari geometry score"},
    {"id": 8, "agent": "GeometryCritic", "prompt": "Score this blueprint: {'base_shape':'chess_pawn','height':1.44,'base_radius':0.55,'head_radius':0.25,'neck_height':0.3}", "desc": "Pawn geometry score"},
    {"id": 9, "agent": "GeometryCritic", "prompt": "Score this blueprint: {'base_shape':'spaceship','length':25,'width':12,'height':6,'wing_span':18,'engine_count':4}", "desc": "Spaceship geometry score"},
    {"id": 10, "agent": "GeometryCritic", "prompt": "Score this blueprint: {'base_shape':'dragon','length':8,'wingspan':12,'height':3.5,'tail_length':3,'neck_length':2}", "desc": "Dragon geometry score"},

    # DesignEnhancer tasks (11-14)
    {"id": 11, "agent": "DesignEnhancer", "prompt": "Enhance this car blueprint: {'base_shape':'sports_car','length':4.7,'width':2.0,'height':1.3,'material':'red_paint','wheel_size':0.35}", "desc": "Sports car design enhancement"},
    {"id": 12, "agent": "DesignEnhancer", "prompt": "Enhance this spaceship: {'base_shape':'starfighter','length':15,'width':8,'height':3,'weapons':2,'engine_type':'ion','color':'dark_gray'}", "desc": "Starfighter design enhancement"},
    {"id": 13, "agent": "DesignEnhancer", "prompt": "Enhance this medieval castle: {'base_shape':'fortress','width':40,'depth':30,'height':25,'tower_count':4,'wall_thickness':3,'material':'stone'}", "desc": "Castle design enhancement"},
    {"id": 14, "agent": "DesignEnhancer", "prompt": "Enhance this cyberpunk bike: {'base_shape':'motorcycle','length':2.2,'height':1.1,'wheel_size':0.6,'engine':'electric','color':'neon_blue'}", "desc": "Cyberpunk bike enhancement"},

    # ProportionAnalyzer tasks (15-18)
    {"id": 15, "agent": "ProportionAnalyzer", "prompt": "Analyze proportions: L=4.92 W=2.08 H=1.35, type=sports_car", "desc": "Ferrari proportion check"},
    {"id": 16, "agent": "ProportionAnalyzer", "prompt": "Analyze proportions: L=0.6 W=0.4 H=1.8, type=humanoid", "desc": "Humanoid proportion check"},
    {"id": 17, "agent": "ProportionAnalyzer", "prompt": "Analyze proportions: L=30 W=25 H=100, type=skyscraper", "desc": "Skyscraper proportion check"},
    {"id": 18, "agent": "ProportionAnalyzer", "prompt": "Analyze proportions: L=2.5 W=1.0 H=1.5, type=horse", "desc": "Horse proportion check"},

    # GameDesigner tasks (19-21)
    {"id": 19, "agent": "GameDesigner", "prompt": "Design a 3D racing game with: 4 tracks, 8 cars, drift mechanics, nitro boost, weather system, AI opponents, split-screen multiplayer", "desc": "Racing game design"},
    {"id": 20, "agent": "GameDesigner", "prompt": "Design a space combat game with: 6 ship types, 3 factions, real-time combat, resource gathering, base building, AI fleet management", "desc": "Space combat game design"},
    {"id": 21, "agent": "GameDesigner", "prompt": "Design a fantasy RPG with: 4 classes, 10 levels, turn-based combat, inventory system, quest system, skill trees, boss battles", "desc": "Fantasy RPG design"},

    # MaterialScientist tasks (22-24)
    {"id": 22, "agent": "MaterialScientist", "prompt": "Enhance this car paint: {'base_color':[0.85,0.05,0.05],'roughness':0.3,'metallic':0.7}", "desc": "Red car paint material"},
    {"id": 23, "agent": "MaterialScientist", "prompt": "Enhance this chrome material: {'base_color':[0.8,0.8,0.9],'roughness':0.05,'metallic':1.0}", "desc": "Chrome material"},
    {"id": 24, "agent": "MaterialScientist", "prompt": "Enhance this organic skin: {'base_color':[0.9,0.7,0.6],'roughness':0.8,'metallic':0.0,'subsurface':0.3}", "desc": "Organic skin material"},

    # EvolutionMutator tasks (25-27)
    {"id": 25, "agent": "EvolutionMutator", "prompt": "Mutate this car: {'base_shape':'sports_car_spline','length':4.92,'width':2.08,'height':1.35,'wheelbase':2.65}", "desc": "Car evolution mutation"},
    {"id": 26, "agent": "EvolutionMutator", "prompt": "Mutate this mech: {'base_shape':'humanoid_mech','height':5.0,'width':2.5,'arm_span':4.0,'weapon_slots':2}", "desc": "Mech evolution mutation"},
    {"id": 27, "agent": "EvolutionMutator", "prompt": "Mutate this aircraft: {'base_shape':'jet','length':19,'wingspan':13.5,'height':4.5,'engine_count':2,'payload':5000}", "desc": "Aircraft evolution mutation"},

    # QualityScorer tasks (28-30)
    {"id": 28, "agent": "QualityScorer", "prompt": "Score this pipeline output: Generated Ferrari Red with 10548 verts, 15532 tris. Dimensions: 4.92x2.08x1.35m. Sports car spline base.", "desc": "Ferrari quality score"},
    {"id": 29, "agent": "QualityScorer", "prompt": "Score this pipeline output: Generated chess set with 12 pieces + board. Avg 650 verts per piece. Board is 8x2x0.15m.", "desc": "Chess set quality score"},
    {"id": 30, "agent": "QualityScorer", "prompt": "Score this pipeline output: Generated medieval castle with 28000 verts, 42000 tris. Dimensions: 40x30x25m. 4 towers, stone material.", "desc": "Castle quality score"},
]


def groq_call(system_prompt, user_prompt, model="llama-3.1-8b-instant", temp=0.3, max_tokens=2048, retry=1):
    global _last_call_time
    if not GROQ_KEY:
        return None
    body = {
        "model": model, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], "temperature": temp, "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        GROQ_URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json", "User-Agent": UA},
        method="POST"
    )
    with _pace_lock:
        elapsed = time.time() - _last_call_time
        if elapsed < 7.0:
            time.sleep(7.0 - elapsed)
        _last_call_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_text = resp.read()
        raw = json.loads(raw_text)
        content = raw["choices"][0]["message"]["content"].strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = None
        if content.startswith("{"):
            depth = 0
            for i, ch in enumerate(content):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0:
                    result = json.loads(content[:i+1])
                    break
        if result is None:
            result = json.loads(content)
        with call_lock:
            call_log.append({"time": time.time(), "model": model})
        return result
    except Exception as e:
        err_str = str(e)
        if "429" in err_str and retry > 0:
            time.sleep(30)
            return groq_call(system_prompt, user_prompt, model, temp, max_tokens, retry-1)
        print(f"  [Groq] Error on task: {e}", flush=True)
        return None


def gemini_analyze(agent_name, task_desc, task_input, agent_output):
    """Send agent's work to Gemini for analysis/review."""
    if not GEMINI_KEY:
        print("  [Gemini] No API key", flush=True)
        return {"error": "no_gemini_key"}
    prompt = f"""You are an Agent Performance Reviewer. Review this agent's work.

Agent: {agent_name}
Task: {task_desc}
Task Input: {json.dumps(task_input)[:500]}
Agent Output: {json.dumps(agent_output)[:2000]}

Analyze:
1. Did the agent correctly follow its instructions?
2. Is the output valid and well-structured?
3. Quality of the response (1-10 scale)
4. Specific strengths
5. Specific weaknesses or improvements needed
6. Overall performance rating (Poor/Fair/Good/Excellent)

Output as JSON with keys: completeness, validity_score, quality_score, strengths, weaknesses, overall_rating, actionable_feedback."""

    body = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(
        GEMINI_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read())
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = None
        if text.startswith("{"):
            depth = 0
            for i, ch in enumerate(text):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0:
                    result = json.loads(text[:i+1])
                    break
        if result is None:
            result = json.loads(text)
        return result
    except Exception as e:
        print(f"  [Gemini] Error: {e}", flush=True)
        return {"error": str(e)}


def git_commit(message):
    try:
        subprocess.run(["git", "-C", str(REPO), "add", "-A"], capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "commit", "-m", message], capture_output=True)
    except:
        pass


def main():
    print("=" * 60, flush=True)
    print(f"30-AGENT TASK ORCHESTRATOR", flush=True)
    print(f"Groq: {'OK' if GROQ_KEY else 'MISSING'} | Gemini: {'OK' if GEMINI_KEY else 'MISSING'}", flush=True)
    print(f"Tasks: {len(TASKS)} | Agents: {len(AGENT_SYSTEM_PROMPTS)}", flush=True)
    print("=" * 60, flush=True)

    # Save task definitions to repo
    for task in TASKS:
        task_file = TASKS_DIR / f"task_{task['id']:03d}.json"
        task_file.write_text(json.dumps(task, indent=2))
    print(f"\nSaved {len(TASKS)} task definitions to {TASKS_DIR}", flush=True)

    completed = 0
    failed = 0
    task_results = []

    for task in TASKS:
        tid = task["id"]
        agent_name = task["agent"]
        prompt = task["prompt"]
        desc = task["desc"]

        print(f"\n{'─'*50}", flush=True)
        print(f"Task {tid:02d}/30: [{agent_name}] {desc}", flush=True)
        print(f"{'─'*50}", flush=True)

        # Get system prompt for this agent
        system_prompt = AGENT_SYSTEM_PROMPTS.get(agent_name, "You are a helpful AI assistant.")

        # Run the agent (Groq)
        t0 = time.time()
        result = groq_call(system_prompt, prompt)
        elapsed = time.time() - t0

        status = "OK" if result else "FAIL"
        print(f"  Result: {status} ({elapsed:.1f}s)", flush=True)

        # Save raw result
        result_data = {
            "task_id": tid,
            "agent": agent_name,
            "description": desc,
            "input_prompt": prompt,
            "output": result,
            "elapsed_s": round(elapsed, 1),
            "timestamp": time.time(),
        }
        result_file = RESULTS_DIR / f"result_{tid:03d}_{agent_name}.json"
        result_file.write_text(json.dumps(result_data, indent=2))

        # Gemini evaluation
        print(f"  Evaluating with Gemini...", flush=True)
        analysis = gemini_analyze(agent_name, desc, prompt, result)
        analysis_data = {
            "task_id": tid,
            "agent": agent_name,
            "description": desc,
            "analysis": analysis,
            "timestamp": time.time(),
        }
        analysis_file = ANALYSES_DIR / f"analysis_{tid:03d}_{agent_name}.json"
        analysis_file.write_text(json.dumps(analysis_data, indent=2))

        # Print summary
        if analysis and "error" not in analysis:
            rating = analysis.get("overall_rating", "N/A")
            quality = analysis.get("quality_score", "N/A")
            validity = analysis.get("validity_score", "N/A")
            print(f"  Gemini: rating={rating} quality={quality} validity={validity}", flush=True)
        else:
            print(f"  Gemini: {analysis}", flush=True)

        if result:
            completed += 1
        else:
            failed += 1

        task_results.append(result_data)

        # Git commit every 5 tasks
        if tid % 5 == 0:
            git_commit(f"Agent tasks 1-{tid}: {completed} OK, {failed} FAIL")

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE: {completed} OK, {failed} FAIL out of {len(TASKS)} tasks", flush=True)
    print(f"Groq API calls: {len(call_log)}", flush=True)

    # Save master report
    report = {
        "timestamp": time.time(),
        "total_tasks": len(TASKS),
        "completed": completed,
        "failed": failed,
        "groq_calls": len(call_log),
        "results": task_results,
    }
    (REPO / "master_report.json").write_text(json.dumps(report, indent=2))
    git_commit(f"Master report: {completed}/{len(TASKS)} tasks completed")

    print(f"Report saved. Ready to scale to 100 tasks.", flush=True)
    print(f"Results stored in: {REPO}", flush=True)


if __name__ == "__main__":
    main()
