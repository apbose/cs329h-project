"""M4: a small neural network over the lotteries and a person embedding, sampled posterior."""

from __future__ import annotations

import numpy as np

from ..data import ChoiceData
from .latent import LaplaceModel


def question_features(data: ChoiceData) -> np.ndarray:
    """``(Q, 8)`` the outcomes and probabilities of both options (riskier option first)."""
    return np.concatenate([data.x.reshape(len(data.x), -1), data.p.reshape(len(data.p), -1)], axis=1)


class NeuralModel(LaplaceModel):
    """M4: ``s_iq = w_2^T tanh(W_1 [phi_q; u_i] + b_1) + b_2``.

    ``phi_q`` holds the payoffs and probabilities of both options and ``u_i`` is a person
    embedding of ``embed_dim`` numbers, so the network shares one function of the lotteries
    across questions. The penalty ``lam`` on the weights (not the biases) is chosen on
    validation people.

    Because the score is far from linear in ``u_i``, two choices differ from M3: each training
    person gets a Gaussian variational posterior instead of a point estimate, which keeps the
    network from overfitting the training people, and a new person's posterior is held as
    importance weights on 4,000 shared samples, as in M2, instead of a Laplace approximation.

    With ``pretrain=True`` the network is first fitted to synthetic choices: M2 is fitted to the
    same training people, ``n_synthetic`` people are drawn from it and answer every question,
    and the network is fitted to their answers (Bourgin et al. 2019). Fitting on the real people
    then starts from those weights and penalizes the distance to them instead of to zero.

    Args:
        use_rating: Let the prior mean depend on the self-rating.
        pretrain: Pretrain on synthetic M2 choices.
        hidden: Hidden units.
        embed_dim: Size of the person embedding.
        n_synthetic: Synthetic people for pretraining.
        hyper: Fixed hyperparameters (skips validation).
        seed: Seed for initialization and validation splits.
    """

    name = "M4"
    penalized = ("W1p", "W1u", "w2")
    hyper_grid = [{"lam": lam} for lam in (0.1, 1.0, 10.0)]
    pretrain_lam = 0.1
    n_rounds = 3
    max_iter = 300
    fit_method = "vi"
    posterior = "samples"

    def __init__(self, use_rating: bool = True, pretrain: bool = False, hidden: int = 16, embed_dim: int = 2,
                 n_synthetic: int = 2000, hyper: dict | None = None, seed: int = 0):
        super().__init__(use_rating=use_rating, hyper=hyper, seed=seed)
        self.pretrain = pretrain
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.n_synthetic = n_synthetic
        if pretrain:
            self.name = "M4-pre"

    def latent_dim(self, hyper: dict) -> int:
        """Size of the person vector: the embedding size, whatever the hyperparameters."""
        return self.embed_dim

    def _bind_questions(self, data: ChoiceData) -> None:
        self._phi = question_features(data)
        self._n_questions = data.n_questions

    def _init_weights(self, rng, hyper):
        H, F, D = self.hidden, self._phi.shape[1], self.embed_dim
        return {"W1p": rng.standard_normal((H, F)) / np.sqrt(F), "W1u": rng.standard_normal((H, D)) / np.sqrt(D),
                "b1": 0.1 * rng.standard_normal(H), "w2": rng.standard_normal(H) / np.sqrt(H), "b2": np.zeros(1)}

    def _hidden(self, w, U):
        a = (self._phi @ w["W1p"].T)[None] + (U @ w["W1u"].T)[:, None] + w["b1"]
        return np.tanh(a)  # (n, Q, H)

    def _scores(self, w, U):
        return self._hidden(w, U) @ w["w2"] + w["b2"][0]

    def _forward(self, w, U):
        h = self._hidden(w, U)
        return h @ w["w2"] + w["b2"][0], h

    def _backprop(self, w, U, G, cache=None):
        h = self._hidden(w, U) if cache is None else cache
        gA = G[..., None] * w["w2"] * (1 - h**2)  # (n, Q, H)
        gA_q, gA_n = gA.sum(0), gA.sum(1)  # (Q, H), (n, H)
        grads = {
            "W1p": gA_q.T @ self._phi,
            "W1u": gA_n.T @ U,
            "b1": gA_q.sum(0),
            "w2": np.tensordot(G, h, axes=([0, 1], [0, 1])),
            "b2": np.array([G.sum()]),
        }
        return grads, gA_n @ w["W1u"]

    def _jac(self, w, U):
        h = self._hidden(w, U)
        s = h @ w["w2"] + w["b2"][0]
        return s, (w["w2"] * (1 - h**2)) @ w["W1u"]

    def _reparam(self, w, c, L):
        out = dict(w)
        out["W1u"], out["b1"] = w["W1u"] @ L, w["b1"] + w["W1u"] @ c
        return out

    def _pretrain(self, data, rows):
        if not self.pretrain:
            return None, None
        from ..simulate import simulate
        from .cpt import ProspectTheory

        m2 = ProspectTheory(use_rating=self.use_rating, n_rounds=5).fit(data, rows)
        pool = data.rating[rows] if (self.use_rating and data.rating is not None) else None
        syn, _ = simulate(m2, data, self.n_synthetic, np.random.default_rng(self.seed + 7), rating_pool=pool)
        self._fit_population(syn, np.arange(syn.n_people), {"lam": self.pretrain_lam})
        init = {"w": {k: v.copy() for k, v in self.w.items()}, "c": self.params.c, "B": self.params.B,
                "Sigma": self.params.Sigma, "eps": self.params.eps}
        self.bind(data)
        return {k: init["w"][k] for k in self.penalized}, init
