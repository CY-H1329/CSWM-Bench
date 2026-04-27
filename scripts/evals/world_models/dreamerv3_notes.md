## DreamerV3 on CSWM images — what is (not) possible

### Why DreamerV3 cannot be “tested” on arbitrary photos + text actions

DreamerV3 is trained on a specific environment with:
- observation space (usually images, but with known preprocessing)
- **numeric action space** (continuous/discrete vectors)
- dynamics defined by that environment

Your CSWM images (door/cup photos or diagrams) have **no associated environment**
and no defined numeric action space, so DreamerV3 cannot roll out futures in a
meaningful way.

If you try anyway, any failure is dominated by **domain/interface mismatch**,
not by the world-model’s causal capability.

### Correct way to evaluate DreamerV3 with “CSWM-style contrastive pairs”

Use an environment DreamerV3 is trained on (or train it on one):

1) Fix the start state \(s_0\) (or same initial frame)
2) Define two action sequences \(A\) and \(A'\) that differ minimally
3) Get GT rollouts from the simulator
4) Get Dreamer imagined rollouts
5) Measure whether Dreamer **separates** the counterfactual futures:
   - event divergence (collision/goal)
   - state divergence (pose)
   - frame divergence (LPIPS/feature distance)

Good starting envs:
- DMControl (walker/cheetah/cartpole)
- ManiSkill (robot manipulation) if you need “drawer/occlusion”

### If you still want to connect DreamerV3 later

What we need from you:
- which DreamerV3 codebase (Danijar repo path)
- which environment (DMControl task name)
- checkpoint path (trained weights)

Then we can implement an adapter:
- input: start seed + action sequences
- output: imagined rollouts + latent distances

