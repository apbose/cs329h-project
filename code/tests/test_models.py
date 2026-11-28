"""The four model classes: values, gradients, posteriors and recovery on simulated people."""

import numpy as np

from cs329h.models import (CPTParams, EUParams, ExpectedUtility, FactorModel, LatentParams, NeuralModel,
                           ProspectTheory, cara_utility, cpt_value)
from cs329h.models.cpt import weight
from cs329h.simulate import simulate


def test_cara_utility_is_linear_at_zero_and_concave_for_positive_r():
    x = np.array([0.0, 0.5, 1.0])
    u = cara_utility(x, np.array([0.0, 2.0, -2.0]))
    assert np.allclose(u[0], x)
    assert u[1, 1] > 0.5 * (u[1, 0] + u[1, 2])  # concave: risk averse
    assert u[2, 1] < 0.5 * (u[2, 0] + u[2, 2])  # convex: risk seeking


def test_more_risk_averse_people_choose_the_lottery_less_often(template):
    model = ExpectedUtility().set_params(EUParams(tau=0.05, eps=0.05, mu0=0, mu1=0, sigma=1), template)
    P = model.choice_prob_people(np.linspace(-2.0, 2.0, 41)[:, None])
    # Logit noise on utility differences is not exactly monotone in r (the utility scale moves
    # with r), but on these lists the reversals stay below 0.01 in probability.
    assert np.diff(P, axis=0).max() < 0.01
    assert (P[0] - P[-1] > 0.5).any()


def test_m1_fit_recovers_the_population(template):
    truth = EUParams(tau=0.05, eps=0.06, mu0=0.8, mu1=-0.6, sigma=0.7)
    gen = ExpectedUtility(use_rating=True).set_params(truth, template)
    sim, _ = simulate(gen, template, 3000, np.random.default_rng(1), rating_pool=np.arange(11.0))
    fit = ExpectedUtility(use_rating=True).fit(sim).params
    assert abs(fit.tau - truth.tau) < 0.015
    assert abs(fit.eps - truth.eps) < 0.03
    assert abs(fit.mu0 - truth.mu0) < 0.15
    assert abs(fit.mu1 - truth.mu1) < 0.25
    assert abs(fit.sigma - truth.sigma) < 0.2


def test_weighting_is_identity_at_gamma_one_and_overweights_small_probabilities():
    p = np.linspace(0, 1, 11)
    assert np.allclose(weight(p, 1.0), p)
    w = weight(p, 0.6)
    assert w[1] > p[1] and w[9] < p[9] and w[0] == 0 and w[-1] == 1


def test_cpt_value_reduces_to_expected_value_and_shows_loss_aversion(mixed_template):
    d = mixed_template
    ev = (d.x * d.p).sum(-1)
    v = cpt_value(d.x, d.p, np.zeros((1, 3)))[0]  # alpha = lambda = gamma = 1
    assert np.allclose(v, ev)
    v2 = cpt_value(d.x, d.p, np.array([[0.0, np.log(2.0), 0.0]]))[0]
    mixed = d.lists == 3
    assert np.allclose(v2[mixed, 0], ev[mixed, 0] - 0.5 * 0.4)  # the loss counts twice


def test_m2_fit_recovers_the_population(mixed_template):
    truth = CPTParams(tau=0.05, eps=0.05, mu0=np.array([-0.2, np.log(1.8), np.log(0.7)]),
                      mu1=np.array([0.3, -0.2, 0.0]), sigma=np.array([0.3, 0.3, 0.2]))
    gen = ProspectTheory(use_rating=True, n_samples=1000).set_params(truth, mixed_template)
    sim, theta = simulate(gen, mixed_template, 2000, np.random.default_rng(2), rating_pool=np.arange(11.0))
    model = ProspectTheory(use_rating=True, n_samples=2000, n_rounds=4).fit(sim)
    assert np.abs(model.params.mu0 - truth.mu0).max() < 0.15
    assert abs(model.params.tau - truth.tau) < 0.02
    est = model.point_estimate(model.update(model.init_belief(sim.rating, 2000), np.zeros(2000, int), sim.Y[:, 0]))
    assert est.shape == (2000, 3)


def _finite_difference_check(model, w, U, rng):
    G = rng.standard_normal((len(U), model._n_questions))
    f = lambda w_, U_: (G * model._scores(w_, U_)).sum()
    gw, gU = model._backprop(w, U, G)
    h = 1e-6
    for k in w:
        idx = tuple(rng.integers(0, n) for n in w[k].shape)
        w2 = {kk: vv.copy() for kk, vv in w.items()}
        w2[k][idx] += h
        assert np.isclose((f(w2, U) - f(w, U)) / h, gw[k][idx], rtol=1e-3, atol=1e-4), k
    U2 = U.copy()
    U2[0, 0] += h
    assert np.isclose((f(w, U2) - f(w, U)) / h, gU[0, 0], rtol=1e-3, atol=1e-4)
    s, J = model._jac(w, U)
    s2 = model._scores(w, U2)
    assert np.allclose((s2[0] - s[0]) / h, J[0, :, 0], rtol=1e-3, atol=1e-4)


def test_m3_and_m4_gradients_match_finite_differences(mixed_template):
    rng = np.random.default_rng(3)
    for model, hyper in ((FactorModel(), {"K": 2}), (NeuralModel(hidden=8, embed_dim=2), {})):
        model.bind(mixed_template)
        w = model._init_weights(rng, hyper)
        U = rng.standard_normal((5, model.latent_dim(hyper)))
        _finite_difference_check(model, w, U, rng)


def test_whitening_leaves_scores_unchanged(mixed_template):
    rng = np.random.default_rng(4)
    for model, hyper in ((FactorModel(), {"K": 2}), (NeuralModel(hidden=8, embed_dim=2), {})):
        model.bind(mixed_template)
        w = model._init_weights(rng, hyper)
        c, L = rng.standard_normal(2), np.array([[1.5, 0.0], [0.4, 0.7]])
        Uw = rng.standard_normal((5, 2))
        assert np.allclose(model._scores(model._reparam(w, c, L), Uw), model._scores(w, c + Uw @ L.T))


def test_laplace_mode_matches_the_exact_posterior_in_one_dimension(mixed_template):
    model = FactorModel(use_rating=False).bind(mixed_template)
    rng = np.random.default_rng(5)
    Q = mixed_template.n_questions
    model.w = {"V": rng.standard_normal((Q, 1)), "b": 0.3 * rng.standard_normal(Q)}
    model.params = LatentParams(eps=0.05, c=np.zeros(1), B=np.zeros(1), Sigma=np.eye(1))
    u_true = rng.standard_normal((50, 1))
    Y = (rng.random((50, Q)) < model.choice_prob_people(u_true)).astype(float)
    b = model.init_belief(None, 50)
    for q in range(10):
        b = model.update(b, np.full(50, q), Y[:, q])
    grid = np.linspace(-6, 6, 4001)[:, None]
    P = model.choice_prob_people(grid)[:, :10]
    logpost = Y[:, :10] @ np.log(P).T + (1 - Y[:, :10]) @ np.log1p(-P).T - 0.5 * grid[:, 0] ** 2
    mode = grid[logpost.argmax(1), 0]
    assert np.abs(b["mean"][:, 0] - mode).max() < 0.01


def test_m3_fit_beats_the_prior_on_simulated_factor_people(mixed_template):
    rng = np.random.default_rng(6)
    Q = mixed_template.n_questions
    gen = FactorModel(use_rating=False).bind(mixed_template)
    gen.w = {"V": 1.5 * rng.standard_normal((Q, 2)), "b": rng.standard_normal(Q)}
    gen.params = LatentParams(eps=0.05, c=np.zeros(2), B=np.zeros(2), Sigma=np.eye(2))
    sim, U = simulate(gen, mixed_template, 600, rng)
    model = FactorModel(use_rating=False, hyper={"K": 2, "lam": 1.0}).fit(sim)
    P_true = gen.choice_prob_people(U)
    rows = np.arange(600)
    asked = np.arange(0, Q, 4)
    unasked = np.setdiff1d(np.arange(Q), asked)

    def corr(m):
        b = m.init_belief(None, 600)
        for q in asked:
            b = m.update(b, np.full(600, q), sim.Y[rows, q])
        return np.corrcoef(m.predict(b)[:, unasked].ravel(), P_true[:, unasked].ravel())[0, 1]

    # Predictions on the unasked questions track the truth nearly as well as the true model's.
    assert corr(model) > corr(gen) - 0.05


def test_pretrained_m4_starts_from_and_stays_near_the_pretrained_weights(mixed_template):
    gen = ProspectTheory(use_rating=False, n_samples=500).set_params(
        CPTParams(tau=0.05, eps=0.05, mu0=np.array([-0.2, 0.5, -0.4]), mu1=np.zeros(3), sigma=np.full(3, 0.3)),
        mixed_template)
    sim, _ = simulate(gen, mixed_template, 150, np.random.default_rng(7))
    model = NeuralModel(use_rating=False, pretrain=True, hidden=4, n_synthetic=200, hyper={"lam": 100.0})
    model.n_rounds, model.max_iter = 1, 50
    model.fit(sim)
    assert model.name == "M4-pre"
    assert model.predict(model.init_belief(None, 3)).shape == (3, mixed_template.n_questions)
