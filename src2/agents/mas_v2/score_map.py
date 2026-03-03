"""
ScoreMap: per-category 2D score maps (LLM x Role).

Each benchmark has K categories.  For every category there is a
|ROLES| x |LLMS| matrix of scores (unbounded).

Step 0  -> random agent selection (per-role independent).
Step >0 -> argmax selection per role (ties broken by first occurrence).
Duplicates across roles are allowed.
"""
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import SPECIALIST_LLMS, ROLES, INITIAL_SCORE


class ScoreMap:
    """Category-specific LLM x Role score maps."""

    def __init__(
        self,
        categories: List[str],
        llms: Optional[List[str]] = None,
        roles: Optional[List[str]] = None,
        initial_score: float = INITIAL_SCORE,
        seed: int = 42,
    ):
        self.categories = list(categories)
        self.llms = list(llms or SPECIALIST_LLMS)
        self.roles = list(roles or ROLES)
        self.initial_score = initial_score
        self.rng = random.Random(seed)

        # _maps[category][role][llm] = float
        self._maps: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cat in self.categories:
            self._maps[cat] = {
                role: {llm: initial_score for llm in self.llms}
                for role in self.roles
            }

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------
    def select_agents(
        self, category: str, step: int,
    ) -> List[Tuple[str, str]]:
        """Return [(role, llm), ...] for each role.

        step == 0 -> per-role independent random choice.
        step >  0 -> per-role argmax (duplicates allowed).
        """
        if category not in self._maps:
            category = self.categories[0]

        assignments: List[Tuple[str, str]] = []
        for role in self.roles:
            if step == 0:
                llm = self.rng.choice(self.llms)
            else:
                scores = self._maps[category][role]
                llm = max(scores, key=scores.get)
            assignments.append((role, llm))
        return assignments

    # ------------------------------------------------------------------
    # Score access / mutation
    # ------------------------------------------------------------------
    def get_score(self, category: str, role: str, llm: str) -> float:
        return self._maps.get(category, {}).get(role, {}).get(
            llm, self.initial_score
        )

    def set_score(self, category: str, role: str, llm: str, value: float):
        if category in self._maps and role in self._maps[category]:
            self._maps[category][role][llm] = value  # no clamp (unbounded)

    def get_category_map(self, category: str) -> Optional[Dict[str, Dict[str, float]]]:
        return self._maps.get(category)

    def get_all_maps(self) -> Dict:
        return copy.deepcopy(self._maps)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str):
        data = {
            "categories": self.categories,
            "llms": self.llms,
            "roles": self.roles,
            "initial_score": self.initial_score,
            "maps": self._maps,
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> "ScoreMap":
        data = json.loads(Path(path).read_text())
        obj = cls(
            categories=data["categories"],
            llms=data["llms"],
            roles=data["roles"],
            initial_score=data["initial_score"],
        )
        obj._maps = data["maps"]
        return obj

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------
    def to_dataframe(self, category: str):
        """Return pandas DataFrame (rows=LLMs, columns=Roles)."""
        import pandas as pd

        rows = {}
        cat_map = self._maps.get(category, {})
        for llm in self.llms:
            rows[llm] = {
                role: cat_map.get(role, {}).get(llm, self.initial_score)
                for role in self.roles
            }
        return pd.DataFrame.from_dict(rows, orient="index", columns=self.roles)

    _ROLE_DISPLAY = {
        "direct_visual_heuristic": "heuristic",
        "explicit_3d_representation": "3D representation",
        "scene_graph_construction": "2D scene graph",
    }

    def to_markdown_list(self, category: str) -> str:
        """Agent → Role: score list format (no table)."""
        cat_map = self._maps.get(category, {})
        lines = []
        for llm in self.llms:
            lines.append("**{}**".format(llm))
            for role in self.roles:
                s = cat_map.get(role, {}).get(llm, self.initial_score)
                rname = self._ROLE_DISPLAY.get(role, role)
                lines.append("  - {}  `{:.3f}`".format(rname, s))
            lines.append("")
        return "\n".join(lines).strip()

    def to_markdown_all_categories(self) -> str:
        """Full score map as markdown (all categories)."""
        lines = []
        for cat in self.categories:
            lines.append("#### `{}`".format(cat))
            lines.append("")
            lines.append(self.to_markdown_list(cat))
            lines.append("")
        return "\n".join(lines).strip()

    def to_pretty_table(self, category: Optional[str] = None) -> str:
        """ASCII table: rows=LLMs, columns=Roles. category=None -> all categories."""
        lines = []
        cats = [category] if category and category in self._maps else self.categories
        for cat in cats:
            cat_map = self._maps.get(cat, {})
            col_names = [self._ROLE_DISPLAY.get(r, r[:10]) for r in self.roles]
            col_widths = [max(len(cn), 10) for cn in col_names]  # fit "0.5234"
            agent_width = max((len(llm) for llm in self.llms), default=10)
            agent_width = max(agent_width, 6)

            sep_len = agent_width + 3 + sum(w + 3 for w in col_widths)
            lines.append("")
            lines.append("Category: {}".format(cat))
            lines.append("-" * sep_len)
            header = "Agent".ljust(agent_width)
            for i, cn in enumerate(col_names):
                header += " | " + cn[:col_widths[i]].center(col_widths[i])
            lines.append(header)
            lines.append("-" * sep_len)
            for llm in self.llms:
                row = llm[:agent_width].ljust(agent_width)
                for role, w in zip(self.roles, col_widths):
                    s = cat_map.get(role, {}).get(llm, self.initial_score)
                    row += " | " + ("{:.4f}".format(s)).rjust(w)
                lines.append(row)
            lines.append("")
        return "\n".join(lines).strip()

    def __repr__(self) -> str:
        return (
            f"ScoreMap(categories={len(self.categories)}, "
            f"llms={len(self.llms)}, roles={len(self.roles)})"
        )
