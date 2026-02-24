# Tool Architecture for Specialist Agents

## Overview

Specialist agents (`explicit_3d_representation`, `scene_graph_construction`) use **external tools** to augment their reasoning. This doc describes the current architecture and how to extend it.

---

## Current: C-Hybrid (Pre-compute + Inject)

**Flow:**
```
Pipeline                    Specialist Agent
   |                              |
   | 1. Run tool (depth/scene)    |
   |    get_depth_summary(img)    |
   |    get_scene_graph_summary   |
   |         ↓                    |
   | 2. Inject tool_output        |
   |    into prompt               |
   |         ↓                    |
   | 3. specialist_generate(      |
   |      llm, image, prompt      |  ← Agent receives image + prompt
   |    )                         |    (prompt already contains tool data)
   |         ↓                    |
   | 4. Parse answer              |
```

**Pros:**
- Simple: works with any VLM (no function-calling API needed)
- Single forward pass per specialist
- Tool runs once per role type (cached in `tool_output_cache`)

**Cons:**
- Agent does NOT "call" tools—it receives pre-computed data
- No dynamic tool selection (e.g. "I need more depth detail")
- Tool always runs even if the question doesn't need it (e.g. Count)

**Code:** `src2/agents/mas_v2/pipeline.py` lines 131–146

---

## Alternative: Agent-Calling (Tool Use Loop)

If you want the agent to **invoke tools dynamically**:

**Flow:**
```
Agent generates → Parser checks for tool call → Run tool → Inject result → Agent generates again (loop)
```

**Implementation options:**

### Option A: Structured output (tool request in text)

1. Prompt the agent: "If you need depth data, output `[CALL: depth_tool]` and wait."
2. Parse agent output for `[CALL: depth_tool]` or similar.
3. If found: run `get_depth_summary(image)`, append to prompt, call agent again.
4. Max iterations (e.g. 2) to avoid infinite loops.

```python
# Pseudocode
for _ in range(max_tool_calls):
    output = specialist_generate(llm, image, prompt)
    if "[CALL: depth_tool]" in output:
        tool_result = get_depth_summary(image)
        prompt += f"\n\n## Tool Result\n{tool_result}\n\nContinue reasoning."
    else:
        break  # No more tool calls
```

### Option B: Function calling API (if model supports it)

Models like GPT-4V, Claude, or Qwen3 with tool-use support can output structured tool calls. You would:

1. Define tool schema (e.g. `depth_estimation(image) -> str`).
2. Pass tools to the model; it returns `tool_calls` in the response.
3. Execute the tool, append result to messages, call model again.

**Requirement:** The specialist LLM (Qwen3-VL-4B) must support tool/function calling. Check the model's API.

### Option C: Two-phase prompt

1. **Phase 1:** "Do you need depth data to answer? Reply YES or NO."
2. If YES: run tool, build full prompt with tool output.
3. **Phase 2:** "Here is the depth data. Answer the question."

This adds latency (2 calls) but avoids parsing tool-call tokens.

---

## Recommendation

- **Keep C-hybrid** for now: it's simple and the tools (depth, scene graph) are cheap to run. Pre-computing them for `explicit_3d` and `scene_graph` roles is fine.
- **Improve ROLE prompting** so the agent uses the injected data effectively (see `prompts.py`).
- **Consider Option A** only if you add expensive tools (e.g. 3D reconstruction) that shouldn't run for every question.

---

## Tool Output Formats

| Tool | Output | Used by |
|------|--------|---------|
| `get_depth_summary(image)` | 9 regions, closer/farther ordering | `explicit_3d_representation` |
| `get_scene_graph_summary(image)` | Objects + pairwise relations | `scene_graph_construction` |

See `src2/tools/depth.py` and `src2/tools/scene_graph.py` for exact formats.
