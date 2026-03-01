"""
Spatial_AOMAS config — Trust Score hyperparameters.
"""
# Phase 1: Hard reward (Case 2)
KAPPA = 1.0  # 추가 패널티 강도 (δ_i에 곱함)

# Phase 2: Category scaling
T_SCALE = 10.0  # φ(N_c) = 1 - exp(-N_c/T), T가 크면 천천히 수렴
GAMMA = 0.1  # s += γ·R̃

# Phase 3: Credibility (EMA)
LAMBDA_F = 0.3  # 단기 EMA 민감도 (f 업데이트)
LAMBDA_G = 0.1  # 장기 EMA 민감도 (g 업데이트)
MU = 0.5  # f vs g 비중 (μ·f + (1-μ)·g)

# 초기값
INITIAL_SCORE = 0.5
INITIAL_N_PLUS = 0.5  # Beta prior
INITIAL_N_MINUS = 0.5
