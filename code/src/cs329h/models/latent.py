"""Models with a continuous person vector and a Laplace posterior (M3 and M4).

A person's vector ``u_i`` (``d`` numbers) has the population prior ``N(c + B z_i, Sigma)``,
``z_i`` the centered self-rating (``B = 0`` without it). The score ``s_iq = f(u_i, q; w)``
depends on population weights ``w`` that a subclass defines.

**Posterior for a new person** (population held fixed). The Laplace approximation
``N(m_i, C_i)``: ``m_i`` is the posterior mode, found by Gauss--Newton steps with step halving,
and ``C_i^{-1} = Sigma^{-1} + sum_q F_iq J_iq J_iq^T`` with ``J_iq = ds_iq / du_i`` and
``F_iq = (dp/ds)^2 / (p (1 - p))`` the Fisher information of one answer. Predictions use the
linearized score, ``s_iq ~ N(s_iq(m_i), J_iq^T C_i J_iq)`` (Immer et al. 2021), and
expectations over it are taken by 20-point Gauss--Hermite quadrature, which is exact enough that
expected information gain needs no sampling.

**Fitting** (Laplace EM on training people). Each round maximizes the joint penalized
likelihood of ``(w, u_1..u_n, eps)`` with the prior held fixed, by L-BFGS with analytic
gradients; the penalty is ``lam * ||w - w_anchor||^2`` on the subclass's penalized weights
(``w_anchor = 0`` unless the weights were pretrained). Then each person's Laplace covariance is
computed at the mode and the prior is updated in closed form: ``(c, B)`` by least squares of
the modes on ``[1, z]`` and ``Sigma`` as the mean of covariance plus residual outer product.
The model is then re-expressed in whitened coordinates ``u = c + L u'`` (``L L^T = Sigma``),
which changes no score, so the fitted prior is always ``N(B z, I)``; without this step the
penalty could be dodged by shrinking ``w`` and inflating ``Sigma``.

Two options serve nonlinear scores (M4): ``fit_method = "vi"`` replaces each training person's
point ``u_i`` by a Gaussian variational posterior (mean and diagonal spread, expectation over
fixed antithetic draws), and ``posterior = "samples"`` holds a new person's posterior as
importance weights on shared samples (as in M2) instead of the Laplace approximation.
Hyperparameters (``lam``, and for M3 the rank) are chosen on validation people: a fifth of the
training people, each with two held-out price lists, scored by their mean held-out log loss over
``k = 0 .. 10`` answers in a random order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..data import ChoiceData
from .base import SupportModel, entropy, rating_z

GH_X, GH_W = np.polynomial.hermite.hermgauss(20)
GH_X, GH_W = GH_X * np.sqrt(2.0), GH_W / np.sqrt(np.pi)  # nodes and weights for N(0, 1)


def lapse_parts(s: np.ndarray, eps: float):
    """Choice probability ``p``, its derivative ``dp/ds`` and ``sigmoid(s)``."""
    sig = expit(s)
    p = np.clip(eps / 2 + (1 - eps) * sig, 1e-9, 1 - 1e-9)
    return p, (1 - eps) * sig * (1 - sig), sig


@dataclass
class LatentParams:
    """Population parameters of a :class:`LaplaceModel` (weights kept on the model)."""

    eps: float
    c: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    hyper: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """The parameters as plain Python values, for JSON."""
        return {"eps": float(self.eps), "c": self.c.tolist(), "B": self.B.tolist(),
                "Sigma": self.Sigma.tolist(), "hyper": dict(self.hyper)}


class LaplaceModel:
    """Base class: belief, posterior and fitting shared by M3 and M4.

    Subclasses define ``hyper_grid`` (candidate hyperparameters), ``latent_dim(hyper)``,
    ``penalized`` (names of penalized weights), ``_init_weights``, ``_scores``, ``_backprop``
    and ``_jac``, and set up question features in ``_bind_questions``.
    """

    name = "latent"
    hyper_grid: list[dict] = []
    penalized: tuple[str, ...] = ()
    n_rounds = 4
    max_iter = 400
    newton_iters = 8
    fit_method = "map"  # or "vi"
    n_vi_draws = 4
    posterior = "laplace"  # or "samples"
    n_support = 4000
    n_epig = 256

    def __init__(self, use_rating: bool = True, hyper: dict | None = None, seed: int = 0):
        self.use_rating = use_rating
        self.hyper = hyper
        self.seed = seed
        self.params: LatentParams | None = None
        self.w: dict | None = None

    def latent_dim(self, hyper: dict) -> int:
        """Size of the person vector ``u_i`` under hyperparameters ``hyper``."""
        raise NotImplementedError

    def _bind_questions(self, data: ChoiceData) -> None:
        raise NotImplementedError

    def _init_weights(self, rng: np.random.Generator, hyper: dict) -> dict:
        raise NotImplementedError

    def _scores(self, w: dict, U: np.ndarray) -> np.ndarray:
        """``(n, Q)`` scores."""
        raise NotImplementedError

    def _forward(self, w: dict, U: np.ndarray):
        """Scores and anything ``_backprop`` can reuse."""
        return self._scores(w, U), None

    def _backprop(self, w: dict, U: np.ndarray, G: np.ndarray, cache=None):
        """Gradients of ``sum(G * scores)`` with respect to the weights (dict) and ``U``."""
        raise NotImplementedError

    def _jac(self, w: dict, U: np.ndarray):
        """Scores ``(n, Q)`` and their Jacobian ``(n, Q, d)`` with respect to ``u``."""
        raise NotImplementedError

    def _reparam(self, w: dict, c: np.ndarray, L: np.ndarray) -> dict:
        """Weights giving the same scores for ``u'`` as ``w`` gives for ``u = c + L u'``."""
        raise NotImplementedError

    @property
    def d(self) -> int:
        """Size of the fitted person vector."""
        return len(self.params.c)

    def bind(self, data: ChoiceData):
        """Attach the questions of ``data``.

        With ``posterior = "samples"`` and a fitted model, also redraw the shared samples.
        """
        self._bind_questions(data)
        if self.posterior == "samples" and self.params is not None:
            self._build_support()
        return self

    def _prior_mean(self, rating, n) -> np.ndarray:
        z = rating_z(rating if self.use_rating else None, n)
        return self.params.c[None] + z[:, None] * self.params.B[None]

    def sample_people(self, rating, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, d)`` person vectors drawn from the population prior."""
        L = np.linalg.cholesky(self.params.Sigma)
        return self._prior_mean(rating, n) + rng.standard_normal((n, self.d)) @ L.T

    def choice_prob_people(self, theta: np.ndarray) -> np.ndarray:
        """``(n, Q)`` choice probabilities for people with vectors ``theta`` (``(n, d)``)."""
        return lapse_parts(self._scores(self.w, theta), self.params.eps)[0]

    def _build_support(self) -> None:
        """Shared samples from a broad normal around the prior (``posterior = "samples"``)."""
        rng = np.random.default_rng(self.seed + 11)
        sd = 1.5 * np.sqrt(np.diag(self.params.Sigma) + 0.25 * self.params.B**2)
        S = self.params.c + sd * rng.standard_normal((self.n_support, self.d))
        self.support_values = S
        self._log_q = (-0.5 * ((S - self.params.c) / sd) ** 2).sum(1) - np.log(sd).sum()
        self._P = self.choice_prob_people(S)

    def init_belief(self, rating, n: int):
        """Belief for ``n`` new people before any answer.

        A dict of ``(n, ...)`` arrays (prior mean ``m0``, which questions were asked and the answers,
        posterior ``mean`` and ``cov``, and the score mean ``s`` and variance ``v`` per question), or
        ``(n, S)`` log importance weights when ``posterior = "samples"``.
        """
        n = len(rating) if rating is not None else n
        if self.posterior == "samples":
            m0 = self._prior_mean(rating, n)
            P0 = np.linalg.inv(self.params.Sigma)
            r = self.support_values[None] - m0[:, None]
            return -0.5 * np.einsum("nsd,de,nse->ns", r, P0, r) - self._log_q[None]
        Q = self._n_questions
        m0 = self._prior_mean(rating, n)
        b = {"m0": m0, "ask": np.zeros((n, Q)), "y": np.zeros((n, Q)), "mean": m0.copy()}
        return self._laplace(b, iters=0)

    def update(self, belief, q: np.ndarray, y: np.ndarray):
        """Add answer ``y[i]`` to question ``q[i]`` and refit the posterior; see ``SupportModel.update``."""
        if self.posterior == "samples":
            return SupportModel.update(self, belief, q, y)
        rows = np.arange(len(q))
        b = dict(belief)
        b["ask"], b["y"] = belief["ask"].copy(), belief["y"].copy()
        b["ask"][rows, q] = 1.0
        b["y"][rows, q] = y
        return self._laplace(b, iters=self.newton_iters)

    def _log_post(self, b, U, P0):
        p = lapse_parts(self._scores(self.w, U), self.params.eps)[0]
        ll = (b["ask"] * (b["y"] * np.log(p) + (1 - b["y"]) * np.log1p(-p))).sum(1)
        r = U - b["m0"]
        return ll - 0.5 * np.einsum("nd,de,ne->n", r, P0, r)

    def _laplace(self, b: dict, iters: int) -> dict:
        """Posterior mode and covariance (in place on the copy ``b``), from ``b["mean"]``."""
        P0 = np.linalg.inv(self.params.Sigma)
        eps = self.params.eps
        U = b["mean"].copy()
        for _ in range(iters):
            s, J = self._jac(self.w, U)
            p, dp, _ = lapse_parts(s, eps)
            g_s = b["ask"] * (b["y"] - p) * dp / (p * (1 - p))
            F = b["ask"] * dp**2 / (p * (1 - p))
            grad = np.einsum("nq,nqd->nd", g_s, J) - (U - b["m0"]) @ P0
            H = P0[None] + np.einsum("nq,nqd,nqe->nde", F, J, J)
            step = np.linalg.solve(H, grad[..., None])[..., 0]
            if np.abs(step).max() < 1e-6:
                break
            f0 = self._log_post(b, U, P0)
            t = np.ones(len(U))
            for _ in range(6):
                cand = U + t[:, None] * step
                bad = self._log_post(b, cand, P0) < f0 - 1e-10
                if not bad.any():
                    break
                t = np.where(bad, t / 2, t)
            U = np.where(bad[:, None], U, cand)
        s, J = self._jac(self.w, U)
        p, dp, _ = lapse_parts(s, eps)
        F = b["ask"] * dp**2 / (p * (1 - p))
        H = P0[None] + np.einsum("nq,nqd,nqe->nde", F, J, J)
        C = np.linalg.inv(H)
        b["mean"], b["cov"] = U, C
        b["s"], b["v"] = s, np.maximum(np.einsum("nqd,nde,nqe->nq", J, C, J), 0.0)
        return b

    def _quad_sig(self, b: dict) -> np.ndarray:
        """``(G, n, Q)`` sigmoid of the score at the quadrature nodes."""
        return expit(b["s"][None] + np.sqrt(b["v"])[None] * GH_X[:, None, None])

    def predict(self, belief) -> np.ndarray:
        """``(n, Q)`` posterior predictive probability of the riskier option; see the module docstring."""
        if self.posterior == "samples":
            return SupportModel.predict(self, belief)
        eps = self.params.eps
        return eps / 2 + (1 - eps) * np.tensordot(GH_W, self._quad_sig(belief), 1)

    def eig(self, belief) -> np.ndarray:
        """``(n, Q)`` expected information gain of each question (entropy of the mean minus mean entropy)."""
        if self.posterior == "samples":
            return SupportModel.eig(self, belief)
        eps = self.params.eps
        pk = eps / 2 + (1 - eps) * self._quad_sig(belief)
        return entropy(np.tensordot(GH_W, pk, 1)) - np.tensordot(GH_W, entropy(pk), 1)

    @staticmethod
    def take(belief, rows):
        """Belief of the people in ``rows``, for either kind of belief."""
        if not isinstance(belief, dict):
            return belief[rows]
        return {k: v[rows] for k, v in belief.items()}

    @staticmethod
    def put(belief, rows, part):
        """Copy of ``belief`` with the people in ``rows`` replaced by ``part``, for either kind of belief."""
        if not isinstance(belief, dict):
            return SupportModel.put(belief, rows, part)
        out = {k: v.copy() for k, v in belief.items()}
        for k in out:
            out[k][rows] = part[k]
        return out

    def point_estimate(self, belief) -> np.ndarray:
        """``(n, d)`` posterior mean of the person vectors (the Laplace mode, or the weighted sample mean)."""
        if self.posterior == "samples":
            return SupportModel.point_estimate(self, belief)
        return belief["mean"]

    def support_probs(self, belief):
        """Weights ``(n, S)`` and probabilities: shared ``(S, Q)``, or ``(n, S, Q)`` from Laplace draws."""
        if self.posterior == "samples":
            return SupportModel.support_probs(self, belief)
        d = belief["mean"].shape[1]
        half = np.random.default_rng(54321).standard_normal((self.n_epig // 2, d))
        E = np.concatenate([half, -half])
        L = np.linalg.cholesky(belief["cov"])
        U = belief["mean"][:, None] + np.einsum("nde,se->nsd", L, E)
        n, S = U.shape[:2]
        P = self.choice_prob_people(U.reshape(n * S, d)).reshape(n, S, -1)
        return np.full((n, S), 1.0 / S), P

    def fit(self, data: ChoiceData, rows: np.ndarray | None = None):
        """Choose hyperparameters on validation people (unless given), then fit on all of ``rows``."""
        self.bind(data)
        rows = np.arange(data.n_people) if rows is None else np.asarray(rows)
        anchor, init = self._pretrain(data, rows)
        hyper = self.hyper
        if hyper is None:
            hyper = self._choose_hyper(data, rows, anchor, init)
        self._fit_population(data, rows, hyper, anchor, init)
        return self

    def _pretrain(self, data, rows):
        """Anchor and starting weights for fitting (none unless a subclass pretrains)."""
        return None, None

    def _fit_population(self, data: ChoiceData, rows, hyper: dict, anchor=None, init=None) -> None:
        Y = data.Y[rows].astype(float)
        n = len(Y)
        z = rating_z(data.rating[rows] if (self.use_rating and data.rating is not None) else None, n)
        X = np.column_stack([np.ones(n), z]) if self.use_rating else np.ones((n, 1))
        d = self.latent_dim(hyper)
        rng = np.random.default_rng(self.seed)
        w = {k: v.copy() for k, v in init["w"].items()} if init else self._init_weights(rng, hyper)
        anchor_w = anchor if anchor is not None else {k: np.zeros_like(w[k]) for k in self.penalized}
        lam = float(hyper["lam"])
        if init:
            c, B, Sigma, eps = init["c"].copy(), init["B"].copy(), init["Sigma"].copy(), init["eps"]
            if not self.use_rating:
                B = np.zeros(d)
        else:
            c, B, Sigma, eps = np.zeros(d), np.zeros(d), np.eye(d), 0.05
        U = c + z[:, None] * B + 0.3 * rng.standard_normal((n, d))
        vi = self.fit_method == "vi"
        logS = np.full((n, d), np.log(0.3))
        if vi:  # fixed draws for the variational expectation (antithetic pairs)
            half = rng.standard_normal((n, self.n_vi_draws // 2, d))
            E = np.concatenate([half, -half], axis=1)  # (n, M, d)
            Y_rep = np.repeat(Y, E.shape[1], axis=0)
        keys = list(w)
        shapes = [w[k].shape for k in keys]
        sizes = [w[k].size for k in keys]

        def pack(w, U, logS, eps):
            """One vector of the weights, person points, log spreads (VI only) and logit of ``eps``."""
            parts = [w[k].ravel() for k in keys] + [U.ravel()] + ([logS.ravel()] if vi else [])
            return np.concatenate(parts + [[np.log(eps / (1 - eps))]])

        def unpack(v):
            """Inverse of ``pack``: weights, ``U`` and log spreads (each ``(n, d)``), and ``eps``."""
            out, i = {}, 0
            for k, sh, sz in zip(keys, shapes, sizes):
                out[k] = v[i:i + sz].reshape(sh)
                i += sz
            U_ = v[i:i + n * d].reshape(n, d)
            logS_ = v[i + n * d:i + 2 * n * d].reshape(n, d) if vi else logS
            return out, U_, logS_, float(expit(v[-1]))

        def lik_terms(w_, U_, Y_, e_):
            """Negative log likelihood of answers ``Y_`` at person points ``U_``, with lapse rate ``e_``.

            Returns the value and its gradients with respect to the weights (dict), ``U_`` and the logit of
            ``e_``.
            """
            s, cache = self._forward(w_, U_)
            p, dp, sig = lapse_parts(s, e_)
            ll = Y_ * np.log(p) + (1 - Y_) * np.log1p(-p)
            r = (Y_ - p) / (p * (1 - p))
            gw, gU = self._backprop(w_, U_, -r * dp, cache)
            # dp/d(eps) = 1/2 - sigmoid(s), times d(eps)/d(logit eps) = eps (1 - eps).
            ge = -(r * (0.5 - sig)).sum() * e_ * (1 - e_)
            return -ll.sum(), gw, gU, ge

        for _ in range(self.n_rounds):
            P0 = np.linalg.inv(Sigma)
            M = c + z[:, None] * B

            def objective(v):
                """Penalized negative log joint per training person, and its gradient, for L-BFGS.

                Under VI the likelihood is averaged over the fixed draws and the prior term is replaced by the
                expected prior minus the entropy of each person's Gaussian.
                """
                w_, U_, logS_, e_ = unpack(v)
                R = U_ - M
                if vi:
                    S_ = np.exp(logS_)
                    m = E.shape[1]
                    draws = (U_[:, None] + S_[:, None] * E).reshape(n * m, d)
                    nll, gw, gD, ge = lik_terms(w_, draws, Y_rep, e_)
                    nll, ge = nll / m, ge / m
                    gw = {k: g / m for k, g in gw.items()}
                    gD = gD.reshape(n, m, d) / m
                    # Expected prior term minus the entropy of q (both in closed form).
                    prior = 0.5 * (np.einsum("nd,de,ne->", R, P0, R) + (np.diag(P0) * S_**2).sum()) - logS_.sum()
                    gU = gD.sum(1) + R @ P0
                    gS = (gD * E).sum(1) * S_ + np.diag(P0) * S_**2 - 1.0
                else:
                    nll, gw, gU, ge = lik_terms(w_, U_, Y, e_)
                    prior = 0.5 * np.einsum("nd,de,ne->", R, P0, R)
                    gU = gU + R @ P0
                pen = sum(((w_[k] - anchor_w[k]) ** 2).sum() for k in self.penalized)
                for k in self.penalized:
                    gw[k] = gw[k] + 2 * lam * (w_[k] - anchor_w[k])
                g = [gw[k].ravel() for k in keys] + [gU.ravel()] + ([gS.ravel()] if vi else [])
                return (nll + prior + lam * pen) / n, np.concatenate(g + [[ge]]) / n

            res = minimize(objective, pack(w, U, logS, eps), jac=True, method="L-BFGS-B",
                           options={"maxiter": self.max_iter})
            w, U, logS, eps = unpack(res.x)
            self.w = w
            # E step (each person's posterior covariance) and closed-form prior update.
            if vi:
                C = np.exp(2 * logS)[:, :, None] * np.eye(d)[None]
            else:
                s_, J = self._jac(w, U)
                p, dp, _ = lapse_parts(s_, eps)
                F = dp**2 / (p * (1 - p))
                C = np.linalg.inv(P0[None] + np.einsum("nq,nqd,nqe->nde", F, J, J))
            coef, *_ = np.linalg.lstsq(X, U, rcond=None)
            resid = U - X @ coef
            c_hat = coef[0]
            B_hat = coef[1] if self.use_rating else np.zeros(d)
            S_hat = (C + resid[:, :, None] * resid[:, None, :]).mean(0) + 1e-4 * np.eye(d)
            # Re-express the model in whitened coordinates u = c_hat + L u', which leaves every
            # score unchanged and keeps the prior at N(B z, I), so the penalty keeps its meaning.
            L = np.linalg.cholesky(S_hat)
            Linv = np.linalg.inv(L)
            w = self._reparam(w, c_hat, L)
            U = (U - c_hat) @ Linv.T
            logS = 0.5 * np.log(np.einsum("de,nef,df->nd", Linv, C, Linv))
            c, B, Sigma = np.zeros(d), Linv @ B_hat, np.eye(d)
        self.w = w
        self.params = LatentParams(eps=eps, c=c, B=B, Sigma=Sigma, hyper=dict(hyper))
        if self.posterior == "samples":
            self._build_support()

    def _choose_hyper(self, data, rows, anchor, init) -> dict:
        """The hyperparameters in ``hyper_grid`` with the lowest held-out loss on validation people."""
        from .. import select
        from ..splits import make_splits

        rng = np.random.default_rng(self.seed + 1)
        perm = rng.permutation(rows)
        n_val = max(len(rows) // 5, 1)
        val, fit_rows = perm[:n_val], perm[n_val:]
        held = make_splits(n_val, data.n_lists, seed=self.seed + 2).heldout_mask(data.lists)
        order = select.random_orders(~held, 10, np.random.default_rng(self.seed + 3))
        Yv = data.Y[val].astype(float)
        rating = data.rating[val] if (self.use_rating and data.rating is not None) else None
        scores = []
        for hyper in self.hyper_grid:
            self._fit_population(data, fit_rows, hyper, anchor, init)
            belief = self.init_belief(rating, n_val)
            loss, _, _, _ = select.replay_curve(self, belief, Yv, order, held)
            scores.append({"hyper": dict(hyper), "val_loss": float(loss.mean())})
        return min(scores, key=lambda r: r["val_loss"])["hyper"]
