"""M3: low-rank factor model (Koren et al. 2009) with a Laplace posterior."""

from __future__ import annotations

import numpy as np

from ..data import ChoiceData
from .latent import LaplaceModel


class FactorModel(LaplaceModel):
    """M3: ``s_iq = u_i^T v_q + b_q``.

    Each question has a learned vector ``v_q`` (``K`` numbers) and offset ``b_q``; each person a
    vector ``u_i``. The rank ``K`` and the penalty ``lam`` on the ``v_q`` (the offsets are not
    penalized) are chosen on validation people from ``hyper_grid``. The person vectors are
    identified only up to rotation, so recovery is judged by predicted choices, not by ``u_i``.
    """

    name = "M3"
    penalized = ("V",)
    hyper_grid = [{"K": k, "lam": lam} for k in (1, 2, 3) for lam in (1.0, 10.0)]

    def latent_dim(self, hyper: dict) -> int:
        """Size of the person vector: the rank ``K``."""
        return int(hyper["K"])

    def _bind_questions(self, data: ChoiceData) -> None:
        self._n_questions = data.n_questions

    def _init_weights(self, rng, hyper):
        Q, K = self._n_questions, int(hyper["K"])
        return {"V": 0.5 * rng.standard_normal((Q, K)), "b": np.zeros(Q)}

    def _scores(self, w, U):
        return U @ w["V"].T + w["b"][None]

    def _backprop(self, w, U, G, cache=None):
        return {"V": G.T @ U, "b": G.sum(0)}, G @ w["V"]

    def _jac(self, w, U):
        J = np.broadcast_to(w["V"][None], (len(U),) + w["V"].shape)
        return self._scores(w, U), J

    def _reparam(self, w, c, L):
        return {"V": w["V"] @ L, "b": w["b"] + w["V"] @ c}
