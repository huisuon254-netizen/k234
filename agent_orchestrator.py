#!/usr/bin/env python3
"""
100+ Agent Task Orchestrator.
Uses this GitHub repo as a server:
  - context/   → stored knowledge (shape DNA, blueprints, etc.)
  - tasks/     → 100+ task definitions  
  - results/   → agent outputs
  - analyses/  → Gemini evaluations of each agent
  - workflows/ → task chain definitions
  - reports/   → master reports per wave
  - wave_output/ → per-wave aggregated data

Workflow: read context → execute task wave → analyze with Gemini → store → evolve context → repeat
"""
import json, os, sys, time, urllib.request, threading, subprocess, shutil
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent

# ── Paths ──
CONTEXT_DIR    = REPO / "context"
TASKS_DIR      = REPO / "tasks"
RESULTS_DIR    = REPO / "results"
ANALYSES_DIR   = REPO / "analyses"
WORKFLOWS_DIR  = REPO / "workflows"
REPORTS_DIR    = REPO / "reports"
WAVE_DIR       = REPO / "wave_output"
AGENT_LOGS     = REPO / "agent_logs"

# ── Load API keys ──
env_path = Path("/home/rachael/Desktop/pinc_forge_complete/src-tauri/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# ── Rate limiting ──
_last_call_time = 0.0
_last_gemini_time = 0.0
_pace_lock = threading.Lock()
_gemini_pace_lock = threading.Lock()
call_log = []
call_lock = threading.Lock()

# ── AGENT SYSTEM PROMPTS ──

AGENT_PROMPTS = {
    "ShapeDNA": """You are a Shape DNA Extraction Agent. Given 3D mesh data, extract design principles as JSON:
{
  "proportions": {"length_width_ratio": float, "width_height_ratio": float},
  "silhouette": {"is_symmetric": bool, "primary_axis": "x"|"y"|"z"},
  "design_language": [str],
  "key_features": [{"name": str, "position": [float,float,float], "description": str}],
  "style_archetype": "sporty"|"elegant"|"aggressive"|"classic"|"modern",
  "complexity_score": float 0-1
}
Output ONLY valid JSON.""",

    "GeometryCritic": """You are a Geometry Quality Critic. Score 3D geometry as JSON:
{
  "overall_score": float 0-1, "proportion_score": float 0-1,
  "topology_score": float 0-1, "symmetry_score": float 0-1,
  "complexity_score": float 0-1,
  "issues": [{"severity": "critical"|"major"|"minor", "description": str}],
  "improvements": [str]
} Output ONLY valid JSON.""",

    "DesignEnhancer": """You are a Design Language Enhancer. Enhance blueprints by improving proportions, style cues, feature placement, material harmony. Output enhanced JSON.""",

    "ProportionAnalyzer": """You are a Proportion Analyst. Analyze dimensions vs real-world expectations as JSON:
{
  "type_detected": str, "expected_proportions": {"L_W": float, "W_H": float},
  "actual_proportions": {"L_W": float, "W_H": float},
  "match_score": float 0-1,
  "adjustments": {"length": float, "width": float, "height": float},
  "notes": [str]
} Output ONLY valid JSON.""",

    "GameDesigner": """You are a Game System Designer. Design game architecture as JSON:
{
  "game_type": str, "required_assets": [{"name":str,"type":str,"count":int}],
  "rules": [{"name":str,"description":str}],
  "systems": [{"name":str,"components":[str]}],
  "interactions": [{"action":str,"result":str}],
  "state_machine": {"states":[str],"transitions":[{"from":str,"to":str,"trigger":str}]},
  "implementation_steps": [str]
} Output ONLY valid JSON.""",

    "MaterialScientist": """You are a PBR Material Scientist. Enhance materials for realism. Output JSON with: base_color (RGB), roughness (0-1), metallic (0-1), clearcoat, anisotropy. Output ONLY valid JSON.""",

    "EvolutionMutator": """You are an Evolution Mutation Agent. Given a blueprint, create a mutated variant. Mutate ONE parameter significantly. Keep recognizable. Add 'mutation_log' field. Output JSON.""",

    "QualityScorer": """You are a Holistic Quality Scorer. Output:
{
  "holistic_score": float 0-1, "aesthetic_score": float 0-1,
  "functional_score": float 0-1, "innovation_score": float 0-1,
  "summary": str, "recommendations": [str]
} Output ONLY valid JSON.""",

    "TaskPlanner": """You are a Task Planning Agent. Given a goal, decompose it into sub-tasks that can be executed by other agents. Output JSON:
{
  "goal": str, "sub_tasks": [{"id": int, "agent": str, "prompt": str, "depends_on": [int]}],
  "estimated_complexity": "low"|"medium"|"high",
  "required_context": [str]
} Output ONLY valid JSON.""",

    "KnowledgeSynthesizer": """You are a Knowledge Synthesis Agent. Given multiple pieces of agent output, synthesize them into coherent knowledge. Output JSON:
{
  "key_insights": [str], "design_principles": [str],
  "cross_domain_patterns": [str], "recommended_focus": str,
  "knowledge_growth": {"new_principles": int, "refined_principles": int}
} Output ONLY valid JSON."""
}

# ── API CALLS ──

def groq_call(system_prompt, user_prompt, model="llama-3.1-8b-instant", temp=0.3, max_tokens=2048, retry=1):
    global _last_call_time
    if not GROQ_KEY: return None
    body = {"model": model, "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temp, "max_tokens": max_tokens}
    req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode(), headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json","User-Agent":UA}, method="POST")
    with _pace_lock:
        elapsed = time.time() - _last_call_time
        if elapsed < 5.0:
            time.sleep(5.0 - elapsed)
        _last_call_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())
        content = raw["choices"][0]["message"]["content"].strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if content.startswith("{"):
            depth = 0
            for i, ch in enumerate(content):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0:
                    result = json.loads(content[:i+1])
                    break
            else:
                result = json.loads(content)
        else:
            result = json.loads(content)
        with call_lock: call_log.append({"time":time.time(),"model":model})
        return result
    except Exception as e:
        if "429" in str(e) and retry > 0:
            time.sleep(30)
            return groq_call(system_prompt, user_prompt, model, temp, max_tokens, retry-1)
        print(f"  [Groq] Error: {e}", flush=True)
        return None


def gemini_call(prompt):
    global _last_gemini_time
    if not GEMINI_KEY: return {"error":"no_key"}
    with _gemini_pace_lock:
        elapsed = time.time() - _last_gemini_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _last_gemini_time = time.time()
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    req = urllib.request.Request(GEMINI_URL, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read())
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if text.startswith("{"):
            depth = 0
            for i, ch in enumerate(text):
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0:
                    return json.loads(text[:i+1])
        return json.loads(text)
    except Exception as e:
        if "429" in str(e):
            time.sleep(10)
            return gemini_call(prompt)
        print(f"  [Gemini] Error: {e}", flush=True)
        return {"error": str(e)}


def git_save(message):
    subprocess.run(["git","-C",str(REPO),"add","-A"], capture_output=True)
    subprocess.run(["git","-C",str(REPO),"commit","-m",message], capture_output=True)
    subprocess.run(["git","-C",str(REPO),"push"], capture_output=True)


# ── TASK GENERATOR ──

def generate_100_tasks(shape_dna, evolved_knowledge):
    """Generate 105 tasks leveraging existing knowledge for AI growth."""
    sd = shape_dna.get("assets", [])
    ferrari_bp = evolved_knowledge.get("ferrari_blueprints", {})
    chess_sd = [a for a in sd if a.get("type") == "chess_piece"]
    avg_L_W = evolved_knowledge.get("shape_dna", {}).get("avg_L_W", 1.0)
    avg_W_H = evolved_knowledge.get("shape_dna", {}).get("avg_W_H", 1.0)

    tasks = []

    # WAVE 1: Shape DNA Extraction on evolved concepts (tasks 1-15)
    concepts = [
        ("hypercar", "L=5.0 W=2.1 H=1.15 verts=12000 first10: [[2.5,0,1.05],[2.5,0,-1.05],[-2.5,0,1.05],[-2.5,0,-1.05],[2.3,0.25,0.95],[2.3,0.25,-0.95],[-2.3,0.25,0.95],[-2.3,0.25,-0.95],[2.5,0.4,1.0],[2.5,0.4,-1.0]]"),
        ("luxury_yacht", "L=30 W=7 H=12 verts=45000 first10: [[15,0,3.5],[15,0,-3.5],[-15,0,3.5],[-15,0,-3.5],[14,2,3],[-14,2,3],[14,2,-3],[-14,2,-3],[15,8,2.5],[-15,8,2.5]]"),
        ("stealth_bomber", "L=21 W=18.5 H=3.5 verts=18000 first10: [[10.5,0,0],[-10.5,0,0],[9,0.6,4.5],[-9,0.6,4.5],[9,0.6,-4.5],[-9,0.6,-4.5],[6,1.2,0.8],[-6,1.2,0.8],[6,1.2,-0.8],[-6,1.2,-0.8]]"),
        ("mecha_suit", "L=1.2 W=0.8 H=2.5 verts=20000 first10: [[0.6,0,0.4],[-0.6,0,0.4],[0.6,0,-0.4],[-0.6,0,-0.4],[0.4,0.5,0.3],[-0.4,0.5,0.3],[0.4,0.5,-0.3],[-0.4,0.5,-0.3],[0.5,2.3,0.2],[-0.5,2.3,0.2]]"),
        ("space_station", "L=50 W=50 H=30 verts=60000 first10: [[25,0,25],[25,0,-25],[-25,0,25],[-25,0,-25],[25,20,25],[25,20,-25],[-25,20,25],[-25,20,-25],[0,15,0],[10,5,10]]"),
        ("dragon", "L=12 W=16 H=5 verts=35000 first10: [[6,0,0],[-6,0,0],[5,0.5,4],[-5,0.5,4],[5,0.5,-4],[-5,0.5,-4],[3,2,2],[-3,2,2],[3,2,-2],[-3,2,-2]]"),
        ("cyberpunk_skyscraper", "L=25 W=20 H=200 verts=55000 first10: [[-12.5,0,-10],[12.5,0,-10],[12.5,0,10],[-12.5,0,10],[-12.5,200,-10],[12.5,200,-10],[12.5,200,10],[-12.5,200,10],[-10,50,-8],[10,50,-8]]"),
        ("battle_tank", "L=8 W=4 H=3 verts=14000 first10: [[4,0,2],[4,0,-2],[-4,0,2],[-4,0,-2],[3.5,0.8,1.8],[3.5,0.8,-1.8],[-3.5,0.8,1.8],[-3.5,0.8,-1.8],[3,1.5,1],[-3,1.5,1]]"),
        ("alien_creature", "L=2.5 W=1.5 H=3 verts=12000 first10: [[1.25,0,0.75],[-1.25,0,0.75],[1.25,0,-0.75],[-1.25,0,-0.75],[1,0.8,0.5],[-1,0.8,0.5],[1,0.8,-0.5],[-1,0.8,-0.5],[0.5,2.8,0.3],[-0.5,2.8,0.3]]"),
        ("submarine", "L=25 W=4 H=6 verts=22000 first10: [[12.5,0,2],[12.5,0,-2],[-12.5,0,2],[-12.5,0,-2],[12,1.5,1.8],[12,1.5,-1.8],[-12,1.5,1.8],[-12,1.5,-1.8],[10,0.5,0],[-10,0.5,0]]"),
        ("futuristic_train", "L=60 W=3.5 H=4.5 verts=32000 first10: [[30,0,1.75],[30,0,-1.75],[-30,0,1.75],[-30,0,-1.75],[29,1.5,1.5],[29,1.5,-1.5],[-29,1.5,1.5],[-29,1.5,-1.5],[28,0.5,1],[-28,0.5,1]]"),
        ("battle_mech", "L=3 W=2.5 H=6 verts=28000 first10: [[1.5,0,1.25],[-1.5,0,1.25],[1.5,0,-1.25],[-1.5,0,-1.25],[1.2,1.0,1.0],[-1.2,1.0,1.0],[1.2,1.0,-1.0],[-1.2,1.0,-1.0],[1.5,5.5,0.8],[-1.5,5.5,0.8]]"),
        ("solar_sail_ship", "L=40 W=60 H=80 verts=25000 first10: [[20,0,30],[20,0,-30],[-20,0,30],[-20,0,-30],[15,40,25],[-15,40,25],[15,40,-25],[-15,40,-25],[10,60,0],[-10,60,0]]"),
        ("underwater_city", "L=100 W=100 H=50 verts=80000 first10: [[50,0,50],[50,0,-50],[-50,0,50],[-50,0,-50],[50,30,50],[50,30,-50],[-50,30,50],[-50,30,-50],[40,50,40],[-40,50,40]]"),
        ("flying_fortress", "L=80 W=40 H=30 verts=70000 first10: [[40,0,20],[40,0,-20],[-40,0,20],[-40,0,-20],[38,10,18],[38,10,-18],[-38,10,18],[-38,10,-18],[35,25,10],[-35,25,10]]"),
    ]
    for i, (name, data) in enumerate(concepts, 1):
        tasks.append({"id": i, "agent": "ShapeDNA", "prompt": f"Analyze a {name}: {data}", "desc": f"{name} DNA extraction", "wave": 1})

    # WAVE 2: GeometryCritic blueprints (tasks 16-30)
    blueprints = [
        ("evolved_hypercar", {"base_shape":"hypercar_spline","length":5.0,"width":2.1,"height":1.15,"wheelbase":2.8,"ground_clearance":0.10,"downforce_ratio":0.45}),
        ("luxury_airship", {"base_shape":"zeppelin","length":120,"diameter":18,"fin_count":4,"cabin_length":40,"cabin_width":8}),
        ("orbital_station", {"base_shape":"ring_station","ring_diameter":80,"hub_diameter":15,"spoke_count":6,"panel_count":24}),
        ("combat_drone", {"base_shape":"quadcopter","arm_span":2.4,"body_length":0.8,"rotor_diameter":0.6,"payload_kg":15}),
        ("medieval_galleon", {"base_shape":"sailing_ship","length":45,"beam":12,"draft":5,"mast_count":3,"deck_count":2}),
        ("cyberpunk_bike", {"base_shape":"motorcycle","length":2.4,"height":1.0,"wheel_size":0.55,"engine":"fusion","armor_rating":3}),
        ("exploration_rover", {"base_shape":"mars_rover","length":3.0,"width":2.5,"height":2.2,"wheel_count":6,"solar_panel_area":4.5}),
        ("stealth_corvette", {"base_shape":"warship","length":85,"beam":12,"draft":4.5,"speed_knots":35,"radar_cross_section":0.5}),
        ("hover_tank", {"base_shape":"hovercraft","length":9,"width":5,"height":2.8,"hover_height":0.5,"armor_thickness":0.12}),
        ("arcology_dome", {"base_shape":"geodesic","diameter":200,"height":100,"layer_count":3,"population":50000}),
        ("fighter_mech", {"base_shape":"humanoid_mech","height":7.0,"width":3.5,"arm_span":5.0,"weapon_slots":4,"speed_kmh":120}),
        ("cargo_drone", {"base_shape":"heavy_lifter","length":5.0,"width":6.0,"height":2.0,"payload_kg":2000,"range_km":800}),
        ("quantum_sub", {"base_shape":"submersible","length":30,"diameter":5,"depth_rating":6000,"crew":8,"endurance_days":45}),
        ("space_elevator", {"base_shape":"tether_station","height":36000,"base_diameter":50,"cable_diameter":0.5,"station_count":4}),
        ("battle_cruiser", {"base_shape":"capital_ship","length":400,"beam":60,"height":80,"fighter_bay":24,"shield_rating":0.95}),
    ]
    for i, (name, bp) in enumerate(blueprints, 16):
        tasks.append({"id": i, "agent": "GeometryCritic", "prompt": f"Score this {name} blueprint: {json.dumps(bp)}", "desc": f"{name} geometry score", "wave": 2})

    # WAVE 3: DesignEnhancer improvements (tasks 31-45)
    designs = [
        ("evolved_ferrari_red", f"Enhance this Ferrari: {json.dumps(ferrari_bp.get('red', {}))}" if ferrari_bp.get('red') else "Enhance this Ferrari: base_shape=sports_car length=4.92 width=2.08 height=1.35 material=red_paint"),
        ("evolved_ferrari_yellow", f"Enhance this Ferrari: {json.dumps(ferrari_bp.get('yellow', {}))}" if ferrari_bp.get('yellow') else "Enhance this Ferrari: base_shape=sports_car length=4.92 width=2.08 height=1.35 material=yellow_paint"),
        ("combat_mech", "Enhance this mech: base_shape=humanoid_mech height=5.5 width=3.0 arm_span=4.5 weapons=2 armor=composite color=battle_gray"),
        ("space_colony", "Enhance this colony: base_shape=toroidal diameter=500 ring_width=30 population=10000 gravity=0.8g"),
        ("stealth_fighter_v2", "Enhance this fighter: base_shape=stealth_jet length=19.5 wingspan=14.0 height=4.8 radar_cross_section=0.01 thrust_vectoring=true"),
        ("cyberpunk_neural_link", "Enhance this neural interface: type=brain_computer bandwidth=10tbps latency=0.1ms channels=4096 form_factor=implant"),
        ("quantum_computer", "Enhance this quantum system: qubits=1024 architecture=superconducting error_rate=1e-6 temperature=15mK"),
        ("fusion_reactor", "Enhance this reactor: type=tokamak plasma_temp=150M power=500MW confinement=2.1T efficiency=0.85"),
        ("warp_drive", "Enhance this drive: type=alcubierre warp_factor=3 energy_requirement=1e20J stability=0.92"),
        ("bio_dome_habitat", "Enhance this habitat: type=closed_ecosystem volume=1e6m3 species_count=5000 self_sufficiency=0.95"),
    ]
    for i, (name, prompt) in enumerate(designs, 31):
        tasks.append({"id": i, "agent": "DesignEnhancer", "prompt": prompt, "desc": f"{name} design enhancement", "wave": 3})

    # WAVE 3 continued (41-45)
    more_designs = [
        ("underwater_habitat", "Enhance this habitat: base_shape=subsea_dome depth=500m diameter=50m compartments=8 material=titanium_alloy"),
        ("orbital_elevator", "Enhance this elevator: base_shape=tether_station height=36000km cable_material=graphene counterweight=asteroid"),
        ("ai_core", "Enhance this AI core: architecture=neural_mesh layers=2048 parameters=1e15 training_data=whole_web compute=10exaflops"),
        ("vertiport", "Enhance this vertiport: base_shape=hexagonal pads=12 air_traffic_capacity=500/day energy=renewable"),
        ("lunar_base", "Enhance this base: location=south_pole modules=8 crew=24 power=nuclear_fission radiation_shielding=regolith"),
    ]
    for i, (name, prompt) in enumerate(more_designs, 41):
        tasks.append({"id": i, "agent": "DesignEnhancer", "prompt": prompt, "desc": f"{name} design enhancement", "wave": 3})

    # WAVE 4: ProportionAnalyzer (tasks 46-60)
    prop_checks = [
        ("concept_car", 5.2, 2.2, 1.1, "sports_car"),
        ("sci-fi_tank", 10, 5, 3.5, "armored_vehicle"),
        ("passenger_drone", 2.5, 2.5, 1.0, "vtol_aircraft"),
        ("cargo_sub", 40, 6, 8, "submarine"),
        ("alien_spaceship", 30, 20, 8, "spacecraft"),
        ("giant_mech", 15, 8, 25, "mecha"),
        ("mobile_ fortress", 50, 30, 20, "land_vehicle"),
        ("orbital_habitat", 200, 200, 50, "space_station"),
        ("stealth_cruiser", 150, 18, 12, "warship"),
        ("hover_bike", 2.8, 0.8, 1.0, "personal_vehicle"),
        ("cargo_zeppelin", 150, 25, 30, "airship"),
        ("deep_sea_drill", 15, 10, 25, "industrial_vehicle"),
        ("colony_ship", 500, 100, 150, "generation_ship"),
        ("fighter_drone", 1.5, 1.5, 0.6, "uav"),
        ("planetary_rover", 4, 3, 2.5, "exploration_vehicle"),
    ]
    for i, (name, l, w, h, typ) in enumerate(prop_checks, 46):
        tasks.append({"id": i, "agent": "ProportionAnalyzer", "prompt": f"Analyze proportions: L={l} W={w} H={h}, type={typ}", "desc": f"{name} proportion check", "wave": 4})

    # WAVE 5: GameDesigner complex games (tasks 61-70)
    games = [
        ("galactic_civilization", "Design a 4X space game with: 8 factions, procedural galaxy, tech tree (200+ techs), diplomacy, trade routes, fleet combat, planetary management, espionage"),
        ("dungeon_siege", "Design a strategy RPG with: 6 hero classes, base building, 50 dungeon levels, trap system, loot rarity tiers, skill combos, boss rush mode"),
        ("cyberpunk_hacker", "Design a hacking simulator with: network mapping, ICE breaking, data extraction, trace countermeasures, reputation system, black market, hardware upgrades"),
        ("colony_manager", "Design a colony sim with: resource chains (20 types), citizen needs, research tree, natural disasters, trade routes, alien encounters, terraforming"),
        ("mech_arena", "Design a mech combat game with: 10 chassis, 30 weapon types, customization, team tactics, environmental destruction, ranking system, esports mode"),
        ("starship_crew", "Design a crew management game with: 6 departments, crisis events, skill trees, ship upgrades, alien diplomacy, boarding actions, mutiny system"),
        ("ninja_stealth", "Design a stealth game with: 5 ninja clans, shadow mechanics, parkour, gadget crafting, detection system, boss fights, replay missions"),
        ("underwater_base", "Design a base building game with: deep sea construction, pressure management, oxygen systems, creature encounters, resource extraction, research"),
        ("time_paradox", "Design a time travel game with: branching timelines, paradox resolution, butterfly effects, temporal weapons, historical eras, timeline stabilization"),
        ("ai_dungeon_master", "Design a procedural D&D-style game with: AI narration, dynamic quests, loot generation, character evolution, party management, moral choices"),
    ]
    for i, (name, req) in enumerate(games, 61):
        tasks.append({"id": i, "agent": "GameDesigner", "prompt": req, "desc": f"{name} game design", "wave": 5})

    # WAVE 6: MaterialScientist (tasks 71-80)
    materials = [
        ("ferrari_red_paint", {'base_color':[0.85,0.05,0.05],'roughness':0.15,'metallic':0.85,'clearcoat':1.0}),
        ("liquid_chrome", {'base_color':[0.85,0.85,0.95],'roughness':0.02,'metallic':1.0,'anisotropy':0.8}),
        ("carbon_fiber", {'base_color':[0.15,0.15,0.15],'roughness':0.4,'metallic':0.0,'normal_strength':0.5}),
        ("alien_chitin", {'base_color':[0.1,0.8,0.3],'roughness':0.3,'metallic':0.6,'iridescence':0.7}),
        ("force_field", {'base_color':[0.2,0.5,1.0],'roughness':0.0,'metallic':0.0,'transmission':0.8,'ior':1.5}),
        ("lava_rock", {'base_color':[0.8,0.2,0.05],'roughness':0.9,'metallic':0.1,'emissive':[0.6,0.1,0.0],'emissive_intensity':2.0}),
        ("ice_crystal", {'base_color':[0.8,0.9,1.0],'roughness':0.05,'metallic':0.0,'transmission':0.9,'ior':1.31}),
        ("neon_glass", {'base_color':[0.0,1.0,0.5],'roughness':0.0,'metallic':0.0,'transmission':0.7,'ior':1.52,'emissive':[0.0,0.8,0.4]}),
        ("damascus_steel", {'base_color':[0.5,0.5,0.5],'roughness':0.3,'metallic':1.0,'anisotropy':0.5,'normal_strength':0.3}),
        ("holographic_fabric", {'base_color':[0.9,0.9,0.9],'roughness':0.5,'metallic':0.0,'iridescence':1.0,'iridescence_ior':1.3}),
    ]
    for i, (name, mat) in enumerate(materials, 71):
        tasks.append({"id": i, "agent": "MaterialScientist", "prompt": f"Enhance {name}: {json.dumps(mat)}", "desc": f"{name} material enhancement", "wave": 6})

    # WAVE 7: EvolutionMutator (tasks 81-90)
    bases = [
        ("sports_car", {'base_shape':'hypercar_spline','length':5.0,'width':2.1,'height':1.15,'wheelbase':2.8,'wing_angle':12}),
        ("combat_mech", {'base_shape':'humanoid_mech','height':6.0,'width':3.0,'arm_span':5.0,'weapon_loadout':['plasma_cannon','missile_pod'],'armor_type':'composite'}),
        ("stealth_jet", {'base_shape':'stealth_aircraft','length':19.5,'wingspan':14.0,'height':4.8,'radar_signature':0.01,'thrust':180}),
        ("battle_cruiser", {'base_shape':'capital_ship','length':400,'beam':60,'height':80,'shield_capacity':5000,'weapon_batteries':12}),
        ("dragon_v2", {'base_shape':'western_dragon','length':12,'wingspan':16,'breath_type':'plasma','scale_color':'crimson','age_years':5000}),
        ("space_station", {'base_shape':'ring_station','ring_diameter':100,'hub_diameter':20,'spoke_count':8,'module_count':24,'crew':200}),
        ("cyberdeck", {'base_shape':'neural_interface','bandwidth':5000,'channels':2048,'processing':100,'storage':5000,'form':'portable'}),
        ("hover_tank_v2", {'base_shape':'hover_vehicle','length':9,'width':5,'height':2.5,'max_speed':150,'armor':120,'weapons':['railgun','point_defense']}),
        ("arcology", {'base_shape':'self_sufficient_dome','diameter':300,'height':150,'population':100000,'energy_source':'fusion','agriculture':'hydroponic'}),
        ("warp_ship", {'base_shape':'faster_than_light','length':250,'width':80,'height':60,'warp_factor':5,'range_ly':1000,'crew':500}),
    ]
    for i, (name, bp) in enumerate(bases, 81):
        tasks.append({"id": i, "agent": "EvolutionMutator", "prompt": f"Mutate this {name}: {json.dumps(bp)}", "desc": f"{name} evolution mutation", "wave": 7})

    # WAVE 8: QualityScorer (tasks 91-100)
    quality_items = [
        ("hypercar_pipeline", "Generated hypercar with 12000 verts, 18000 tris. L=5.0 W=2.1 H=1.15. Active aero, hybrid engine, carbon body."),
        ("chess_set_complete", "Generated 12 chess pieces + board. Avg 650 verts/piece. Board 8x2x0.15m. White and black variants."),
        ("space_station_complex", "Generated orbital station with 60000 verts, 90000 tris. Ring diameter 80m, 6 spokes, 24 modules."),
        ("mech_squad", "Generated 4 mech variants: scout, assault, sniper, support. Avg 20000 verts each. Weapon systems integrated."),
        ("dragon_family", "Generated 3 dragon variants: fire, ice, void. Avg 35000 verts. Breath effects, wing membranes, scale textures."),
        ("city_blocks", "Generated 10 city blocks: residential, commercial, industrial, park. Avg 40000 verts/block. Procedural buildings."),
        ("fleet_assets", "Generated 8 ship classes: fighter, corvette, frigate, cruiser, battleship, carrier, dreadnought, titan."),
        ("alien_ecosystem", "Generated 12 alien creature types: herbivore, predator, flying, aquatic. Avg 15000 verts. Procedural rigging."),
        ("weapon_arsenal", "Generated 30 weapon models: ballistic, energy, explosive, melee. Avg 3000 verts. PBR materials applied."),
        ("terraform_modules", "Generated 6 terrain biomes: desert, tundra, jungle, ocean, volcanic, urban. Avg 100000 verts each."),
    ]
    for i, (name, desc) in enumerate(quality_items, 91):
        tasks.append({"id": i, "agent": "QualityScorer", "prompt": f"Score: {desc}", "desc": f"{name} quality score", "wave": 8})

    # WAVE 9: Meta-tasks - TaskPlanner (tasks 101-105)
    meta_goals = [
        "Build a complete metaverse city with 10 districts, 100 buildings, 500 NPCs, day/night cycle, weather, and public transport",
        "Create a procedurally generated alien planet with 5 biomes, 20 flora species, 15 fauna species, caves, and ancient ruins",
        "Design a fully playable MMORPG with 12 classes, 100 levels, 50 dungeons, PvP arenas, crafting, and player housing",
        "Build a real-time strategy game with 4 factions, 200 units, tech trees, resource chains, and AI opponent with 3 difficulty levels",
        "Create a virtual AI training environment with 10 scenarios, reward systems, agent observation, and performance metrics",
    ]
    for i, goal in enumerate(meta_goals, 101):
        tasks.append({"id": i, "agent": "TaskPlanner", "prompt": f"Plan tasks for: {goal}", "desc": f"Meta-plan: {goal[:50]}...", "wave": 9})

    return tasks


def load_context():
    ctx = {"shape_dna": {}, "evolved_knowledge": {}}
    sd_file = CONTEXT_DIR / "shape_dna.json"
    ek_file = CONTEXT_DIR / "evolved_knowledge.json"
    if sd_file.exists(): ctx["shape_dna"] = json.loads(sd_file.read_text())
    if ek_file.exists(): ctx["evolved_knowledge"] = json.loads(ek_file.read_text())
    return ctx


def update_context(task_results):
    """Synthesize new knowledge from completed tasks and update context."""
    ek_file = CONTEXT_DIR / "evolved_knowledge.json"
    ctx = load_context()
    ek = ctx.get("evolved_knowledge", {})
    gen = ek.get("generation_run", 0) + 1

    # Extract insights from completed tasks
    insights = []
    for r in task_results[-50:]:
        out = r.get("output", {})
        if isinstance(out, dict):
            if "overall_score" in out:
                insights.append(f"Geometry score: {out['overall_score']} from {r.get('desc','')}")
            if "holistic_score" in out:
                insights.append(f"Quality score: {out['holistic_score']} from {r.get('desc','')}")
            if "complexity_score" in out:
                insights.append(f"Complexity: {out['complexity_score']} from {r.get('desc','')}")

    new_ek = {
        "timestamp": time.time(),
        "generation_run": gen,
        "total_tasks_completed": len(task_results),
        "latest_insights": insights[-20:],
        "shape_dna": ctx.get("shape_dna", {}).get("extracted_at", 0) if ctx.get("shape_dna") else 0,
    }
    ek_file.write_text(json.dumps(new_ek, indent=2))
    return new_ek


def run_wave(tasks, wave_num, all_results):
    print(f"\n{'='*60}", flush=True)
    print(f"WAVE {wave_num}: {len(tasks)} tasks", flush=True)
    print(f"{'='*60}", flush=True)

    wave_results = []
    for task in tasks:
        tid = task["id"]
        agent_name = task["agent"]
        prompt = task["prompt"]
        desc = task["desc"]

        print(f"\n  Task {tid:03d}/{len(tasks)}: [{agent_name}] {desc}", flush=True)

        system_prompt = AGENT_PROMPTS.get(agent_name, "You are a helpful AI assistant.")
        t0 = time.time()
        result = groq_call(system_prompt, prompt)
        elapsed = time.time() - t0

        result_data = {
            "task_id": tid, "agent": agent_name, "description": desc,
            "input_prompt": prompt, "output": result,
            "elapsed_s": round(elapsed, 1), "timestamp": time.time(),
            "wave": wave_num,
        }
        wave_results.append(result_data)
        all_results.append(result_data)

        # Store result
        (RESULTS_DIR / f"result_{tid:03d}_{agent_name}.json").write_text(json.dumps(result_data, indent=2))

        # Gemini analysis
        print(f"    Gemini analyzing...", flush=True)
        analysis_prompt = f"""Review this agent's work:
Agent: {agent_name}
Task: {desc}
Output: {json.dumps(result)[:2000] if result else 'FAILED'}

Score: completeness (0-10), quality (0-10), innovation (0-10).
Rating: Poor/Fair/Good/Excellent.
Output JSON with: completeness, quality_score, innovation_score, overall_rating, strengths, weaknesses, improvement_tips."""

        analysis = gemini_call(analysis_prompt)
        analysis_data = {
            "task_id": tid, "agent": agent_name, "description": desc,
            "analysis": analysis, "timestamp": time.time(), "wave": wave_num,
        }
        (ANALYSES_DIR / f"analysis_{tid:03d}_{agent_name}.json").write_text(json.dumps(analysis_data, indent=2))

        status = "OK" if result else "FAIL"
        rating = analysis.get("overall_rating", "?") if isinstance(analysis, dict) else "?"
        print(f"    → {status} | Gemini: {rating} | {elapsed:.1f}s", flush=True)

        # Save every 10 tasks
        if tid % 10 == 0:
            git_save(f"Wave {wave_num}: {tid}+ tasks completed")

    return wave_results


def main():
    print("=" * 70, flush=True)
    print(f"100+ AGENT TASK ORCHESTRATOR", flush=True)
    print(f"Groq: {'OK' if GROQ_KEY else 'MISSING'} | Gemini: {'OK' if GEMINI_KEY else 'MISSING'}", flush=True)
    print(f"Repo server: {REPO}", flush=True)
    print("=" * 70, flush=True)

    # Load context
    ctx = load_context()
    print(f"Context: shape_dna={bool(ctx['shape_dna'])} evolved_knowledge={bool(ctx['evolved_knowledge'])}", flush=True)

    # Generate 105 tasks
    tasks = generate_100_tasks(ctx["shape_dna"], ctx["evolved_knowledge"])
    print(f"Generated {len(tasks)} tasks across 9 waves", flush=True)

    # Save all tasks to repo
    for t in tasks:
        (TASKS_DIR / f"task_{t['id']:03d}.json").write_text(json.dumps(t, indent=2))
    print(f"Saved {len(tasks)} task files to {TASKS_DIR}", flush=True)

    all_results = []
    waves = {}
    for t in tasks:
        w = t.get("wave", 1)
        if w not in waves: waves[w] = []
        waves[w].append(t)

    # Process waves sequentially
    for wave_num in sorted(waves.keys()):
        wave_tasks = waves[wave_num]
        wr = run_wave(wave_tasks, wave_num, all_results)
        # Wave report
        ok_count = sum(1 for r in wr if r.get("output"))
        fail_count = len(wr) - ok_count
        report = {
            "wave": wave_num, "tasks": len(wr),
            "completed": ok_count, "failed": fail_count,
            "timestamp": time.time(), "results": wr,
        }
        (WAVE_DIR / f"wave_{wave_num}_report.json").write_text(json.dumps(report, indent=2))
        print(f"\n  Wave {wave_num} done: {ok_count} OK, {fail_count} FAIL", flush=True)

        # Update context/grow knowledge
        update_context(all_results)
        print(f"  Knowledge grown. Total completed: {len(all_results)}", flush=True)

    # Master report
    all_ok = sum(1 for r in all_results if r.get("output"))
    all_fail = len(all_results) - all_ok
    master = {
        "timestamp": time.time(),
        "total_tasks": len(tasks),
        "completed": all_ok,
        "failed": all_fail,
        "groq_calls": len(call_log),
        "waves": sorted(waves.keys()),
        "summary": f"{all_ok}/{len(tasks)} tasks OK across {len(waves)} waves",
    }
    (REPORTS_DIR / "master_report.json").write_text(json.dumps(master, indent=2))
    (REPO / "master_report.json").write_text(json.dumps(master, indent=2))

    git_save(f"Complete: {all_ok}/{len(tasks)} tasks, {len(waves)} waves")
    print(f"\n{'='*70}", flush=True)
    print(f"COMPLETE! {all_ok}/{len(tasks)} tasks OK ({all_fail} FAIL)", flush=True)
    print(f"Groq calls: {len(call_log)}", flush=True)
    print(f"Results: {REPO}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
