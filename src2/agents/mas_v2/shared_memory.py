"""
SharedMemory: per-step cache for specialist agent outputs.

Cleared and rebuilt every step (like a cache register).
The Final Reasoning Agent reads the full contents.
"""
from typing import Dict, List


class SharedMemory:
    """Per-step shared memory between specialist agents and the reasoning agent."""

    def __init__(self):
        self._entries: List[Dict] = []

    def add(self, role: str, llm_name: str, answer: str, reason: str):
        self._entries.append({
            "role": role,
            "llm_name": llm_name,
            "answer": answer,
            "reason": reason,
        })

    def clear(self):
        self._entries = []

    def get_entries(self) -> List[Dict]:
        return list(self._entries)

    def to_prompt_text(self) -> str:
        """Format all entries as text for the Final Reasoning Agent prompt."""
        lines = []
        for i, e in enumerate(self._entries, 1):
            lines.append(f"### Agent {i}: {e['role']} (Model: {e['llm_name']})")
            lines.append(f"- Answer: {e['answer']}")
            lines.append(f"- Reasoning: {e['reason']}")
            lines.append("")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SharedMemory(entries={len(self._entries)})"
