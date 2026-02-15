"""
Correlation service – cross-pair analytics.
>>> WHERE TO PLUG REAL MODELS: replace with realized correlation from
    historical returns, DCC-GARCH, or shrinkage estimators.
"""

import numpy as np
from app.schemas import CorrelationMatrix, ScatterPoint, CorrelationResponse


def generate_correlations(pairs: list[str], seed: int = 42) -> CorrelationResponse:
    rng = np.random.default_rng(seed)
    n = len(pairs)

    # random positive-definite correlation matrix
    A = rng.normal(0, 1, (n, n))
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    matrix = CorrelationMatrix(
        pairs=pairs,
        matrix=[[round(float(corr[i][j]), 4) for j in range(n)] for i in range(n)],
    )

    scatter = []
    for p in pairs:
        scatter.append(ScatterPoint(
            pair=p,
            ann_return=round(float(rng.uniform(2, 18)), 2),
            ann_vol=round(float(rng.uniform(5, 16)), 2),
        ))

    return CorrelationResponse(matrix=matrix, scatter=scatter)
