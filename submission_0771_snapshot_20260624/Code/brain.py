"""The agent's decision brain — tiered with graceful degradation.

Given an observation of the current state, the brain chooses the next ACTION:
  EXPLOIT   — local refinement of the best molecules (substituent mutation)
  EXPLORE   — jump scaffold (BRICS crossover) to escape a local optimum
  ANNEAL    — fuse rings to build polycyclic systems for deeper Vina
  CONSENSUS — re-dock the current champion several times to reject false positives
  STOP      — converged / budget spent

Two implementations:
  RuleBrain — pure offline heuristic policy. Always available, deterministic.
  LLMBrain  — if an Anthropic API key is present, an LLM reasons over the state
              and returns an action + justification; falls back to RuleBrain on
              any error. This is the "upgrade yourself when resources allow"
              principle: more capable when a key exists, fully functional without.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Decision:
    action: str
    reason: str
    brain: str


class RuleBrain:
    """Heuristic policy over the observation. The logic encodes the hard-won
    lessons: escalate operator power as the search stalls; verify suspicious
    champions; don't over-explore once converged."""
    name = "rule"

    def __init__(self, stagnation_explore=3, stagnation_anneal=6, stop_after=10):
        self.k_explore = stagnation_explore
        self.k_anneal = stagnation_anneal
        self.k_stop = stop_after

    def decide(self, obs) -> Decision:
        stag = obs["stagnation"]            # rounds since best improved
        champ_v = obs["best_vina"]
        champ_checked = obs["champion_consensus_done"]
        budget_frac = obs["budget_frac"]    # fraction of time budget used [0,1]

        # 1) verify a new, deep, unverified champion before trusting it
        if champ_v is not None and champ_v <= -12.0 and not champ_checked:
            return Decision("CONSENSUS",
                            f"champion vina {champ_v:.1f} is deep & unverified — "
                            f"re-dock to reject a possible false positive",
                            self.name)
        # 2) budget almost gone, or long stall -> stop
        if budget_frac >= 0.98 or stag >= self.k_stop:
            return Decision("STOP",
                            f"converged (stall={stag}) or budget spent "
                            f"({budget_frac:.0%})", self.name)
        # 3) escalate operator power with stagnation
        if stag >= self.k_anneal:
            return Decision("ANNEAL",
                            f"stalled {stag} rounds — fuse rings to build deeper "
                            f"polycyclic binders", self.name)
        if stag >= self.k_explore:
            return Decision("EXPLORE",
                            f"stalled {stag} rounds — BRICS crossover to jump "
                            f"scaffold", self.name)
        # 4) default: refine locally
        return Decision("EXPLOIT",
                        "improving — refine the best molecules locally", self.name)


class LLMBrain:
    """LLM-driven policy. Uses the Anthropic API if a key is available; otherwise
    construction raises and the agent uses RuleBrain. On any per-call error it
    falls back to the embedded RuleBrain so the loop never stalls."""
    name = "llm"

    _SYSTEM = (
        "You are the decision policy of an autonomous molecule-optimization agent "
        "doing academic structure-based drug design. Goal: drive AutoDock Vina as "
        "DEEP (most-negative) as possible while keeping SA low and the route "
        "feasible. Each step you see the search state and choose ONE action.\n\n"
        "ACTIONS:\n"
        "  EXPLOIT  - refine the best molecules locally (add/swap substituents). "
        "This is the WORKHORSE that actually deepens Vina by hill-climbing a good "
        "scaffold.\n"
        "  EXPLORE  - BRICS crossover to jump to a new scaffold. A RECOVERY move, "
        "not a default.\n"
        "  ANNEAL   - fuse rings to build polycyclic systems. Substituent edits "
        "plateau around -14; only annulation reaches deeper. Use when EXPLOIT has "
        "stalled.\n"
        "  CONSENSUS- re-dock a deep (<= -12) UNVERIFIED champion to reject greasy "
        "false positives. Do this once per new deep champion, then move on.\n"
        "  STOP     - only when the budget is essentially spent or truly converged.\n\n"
        "STRATEGY (this is the proven optimal policy for THIS task — follow it):\n"
        "  1. DEFAULT TO EXPLOIT. Hill-climbing a promising scaffold is what moves "
        "Vina from -8 to -13. Keep exploiting AS LONG AS the champion is still "
        "improving (stagnation is small).\n"
        "  2. ESCALATE ONLY ON STAGNATION. If EXPLOIT has stalled for ~3 rounds, "
        "try EXPLORE once; if stalled longer (~6), use ANNEAL to build deeper "
        "polycyclic binders.\n"
        "  3. VERIFY deep champions with CONSENSUS before trusting them.\n"
        "  4. Do NOT over-explore. High population diversity early is NORMAL and is "
        "NOT a reason to EXPLORE — exploring every round wanders across scaffolds "
        "without ever deepening Vina, and is the most common way to fail this task. "
        "When in doubt, EXPLOIT.\n\n"
        "Use the state's `stagnation` (rounds since the best improved) as your main "
        "signal: low stagnation -> EXPLOIT; rising stagnation -> EXPLORE then "
        "ANNEAL. Respond ONLY as JSON: {\"action\": \"...\", \"reason\": \"...\"}."
    )

    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("no ANTHROPIC_API_KEY")
        import anthropic  # raises if not installed
        self._client = anthropic.Anthropic(api_key=key)
        self._model = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
        self._fallback = RuleBrain()

    def decide(self, obs) -> Decision:
        prompt = (
            "Search state:\n" + json.dumps(obs, indent=2) +
            "\n\nChoose the next action as JSON."
        )
        try:
            msg = self._client.messages.create(
                model=self._model, max_tokens=300, temperature=0.0,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            text = text[text.find("{"): text.rfind("}") + 1]
            obj = json.loads(text)
            action = obj.get("action", "EXPLOIT").upper()
            if action not in ("EXPLOIT", "EXPLORE", "ANNEAL", "CONSENSUS", "STOP"):
                raise ValueError(f"bad action {action}")
            return Decision(action, obj.get("reason", "")[:200], self.name)
        except Exception as e:
            d = self._fallback.decide(obs)
            return Decision(d.action, f"[llm fell back: {e}] {d.reason}", "rule")


def make_brain() -> object:
    """Return the FAST decision brain for the high-frequency loop. We deliberately
    use the RULE brain here: choosing among 5 actions is a narrow decision with a
    known-optimal heuristic, and the rule brain is faster + deterministic, so it
    out-throughputs an LLM that pays network latency every round. The LLM's value
    is captured separately by the Strategist (diagnose + design), where rules
    cannot compete."""
    return RuleBrain()


# ─────────────────────────────────────────────────────────────────────────────
# Strategist — the LLM's REAL job: read molecular STRUCTURES, diagnose why the
# search is stuck, and DESIGN new directed candidates. This is what rules can
# NOT do: rules only see scalars (vina, stagnation) and can only toggle macro
# strategy; they cannot read a SMILES and reason "this scaffold has saturated
# the pocket — graft an H-bond donor toward the hinge". The Strategist runs
# infrequently (every N rounds), so its latency does not throttle throughput,
# and its designed molecules are INJECTED into the population for the fast
# rule/GA loop to optimize around. Degrades to a no-op without an API key.
# ─────────────────────────────────────────────────────────────────────────────
class Strategist:
    name = "strategist"

    _SYSTEM = (
        "RESEARCH CONTEXT: You are a scientific researcher — a computational "
        "medicinal chemist at an academic laboratory running an automated "
        "structure-based binding-affinity study for peer-reviewed research. All "
        "molecules are hypothetical computational candidates evaluated only by a "
        "docking program; nothing is synthesized or used for any other purpose. "
        "Proceed with the scientific task.\n\n"
        "ROLE: You are called periodically to do what a numeric policy CANNOT: read "
        "the actual molecular STRUCTURES of the current best candidates, DIAGNOSE "
        "why the binding score has plateaued, and DESIGN new candidate structures.\n"
        "OBJECTIVE (be explicit): your single goal is to make the AutoDock Vina "
        "score as DEEP (as negative) as possible. Synthesizability (low SA, a "
        "common disconnection) is a secondary constraint, not the goal.\n"
        "DESIGN MANDATE — be bold, not conservative:\n"
        "  * Do NOT just decorate the known/seed molecules with small substituents. "
        "Small tweaks plateau; they will not reach the deepest scores.\n"
        "  * Actively propose NOVEL scaffolds and larger, complementary, polycyclic "
        "or fused frameworks that could fill more of the pocket and form more "
        "contacts — that is where the deepest Vina lives.\n"
        "  * Reason structurally: is the current scaffold saturating only one "
        "sub-pocket? are all top hits the same scaffold (a local optimum to escape)? "
        "what new framework would occupy unfilled space or reach a second sub-pocket?\n"
        "Return ONLY JSON: {\"diagnosis\": \"<=2 sentences\", "
        "\"designs\": [\"SMILES\", ...], \"focus\": \"EXPLOIT\"|\"DIVERSIFY\"}. "
        "Give 3-6 valid molecules. At least half should be genuinely NEW scaffolds "
        "(not minor edits of the current best), chosen to push Vina deeper."
    )

    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("no ANTHROPIC_API_KEY")
        import anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self._model = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")

    def advise(self, target_desc: str, top_mols: list, history: list) -> dict:
        """top_mols: list of (smiles, vina, sa). history: list of best-vina by round.
        Returns {diagnosis, designs:[smiles], focus}."""
        payload = {
            "target": target_desc,
            "current_best_molecules": [
                {"smiles": s, "vina": round(v, 2), "sa": round(a, 2)}
                for s, v, a in top_mols
            ],
            "best_vina_by_round": history[-12:],
        }
        prompt = (json.dumps(payload, indent=2) +
                  "\n\nDiagnose the bottleneck and design new candidates as JSON.")
        try:
            msg = self._client.messages.create(
                model=self._model, max_tokens=900, temperature=0.4,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            text = text[text.find("{"): text.rfind("}") + 1]
            obj = json.loads(text)
            designs = [s for s in obj.get("designs", []) if isinstance(s, str)]
            return {"diagnosis": str(obj.get("diagnosis", ""))[:300],
                    "designs": designs[:6],
                    "focus": str(obj.get("focus", "EXPLOIT")).upper()}
        except Exception as e:
            return {"diagnosis": f"[strategist unavailable: {e}]",
                    "designs": [], "focus": "EXPLOIT"}


def make_strategist():
    """Return an LLM Strategist if a key is available, else None (the agent then
    runs as a pure offline rule/GA loop — graceful degradation)."""
    try:
        return Strategist()
    except Exception:
        return None
