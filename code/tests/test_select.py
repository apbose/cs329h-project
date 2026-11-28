"""Information gain, the selection rules and the replay."""

import numpy as np

from cs329h import select
from cs329h.models import EUParams, ExpectedUtility
from cs329h.models.base import SupportModel
from cs329h.simulate import simulate


def test_question_that_cannot_discriminate_has_zero_gain():
    m = SupportModel()
    m._P = np.array([[0.3, 0.1], [0.3, 0.9]])  # question 0 is the same at both support points
    gain = m.eig(np.log(np.array([[0.5, 0.5]])))
    assert abs(gain[0, 0]) < 1e-12 and gain[0, 1] > 0.1


def setup(template, n=400, seed=0):
    truth = EUParams(tau=0.05, eps=0.05, mu0=0.5, mu1=0.0, sigma=1.0)
    model = ExpectedUtility(use_rating=False).set_params(truth, template)
    sim, r = simulate(model, template, n, np.random.default_rng(seed))
    held = np.zeros_like(sim.Y, bool)
    held[:, sim.lists == 3] = True
    return sim, model, model.init_belief(None, n), held


def test_rules_never_ask_held_out_or_repeated_questions(template):
    sim, model, b, held = setup(template)
    avail = ~held
    for asked in (
        select.adaptive_order(model, b, sim.Y, avail, 6),
        select.random_orders(avail, 6, np.random.default_rng(0)),
        select.original_orders(avail, 6),
    ):
        assert not held[np.arange(len(asked))[:, None], asked].any()
        assert all(len(set(row)) == 6 for row in asked)


def test_adaptive_beats_original_order_with_known_parameters(template):
    sim, model, b, held = setup(template)
    avail = ~held
    Y = sim.Y.astype(float)
    loss_a, *_ = select.replay_curve(model, b, Y, select.adaptive_order(model, b, sim.Y, avail, 4), held)
    loss_o, *_ = select.replay_curve(model, b, Y, select.original_orders(avail, 4), held)
    assert loss_a[:, 0].mean() == loss_o[:, 0].mean()  # same prior before any answer
    assert loss_a[:, 4].mean() < loss_o[:, 4].mean()


def test_greedy_fixed_order_gives_everyone_enough_questions(template):
    sim, model, b, held = setup(template)
    order = select.greedy_fixed_order(model, b, sim.Y.astype(float), held, 5)
    asked = select.first_available(order, ~held, 5)
    assert asked.shape == (len(sim.Y), 5)


def test_predictive_gain_rewards_questions_that_inform_the_targets():
    # Two equally likely types; question 0 separates them, question 1 is pure noise, and
    # question 2 (the target) is what question 0 predicts.
    P = np.array([[0.95, 0.5, 0.95], [0.05, 0.5, 0.05]])
    gain = select.predictive_gain(np.array([0.5, 0.5]), P, np.array([False, False, True]))
    assert gain[0] > 0.3 and abs(gain[1]) < 1e-9


def test_predictive_rule_avoids_held_out_and_repeats_and_beats_original(template):
    sim, model, b, held = setup(template)
    avail = ~held
    asked = select.predictive_order(model, b, sim.Y, avail, 4)
    assert not held[np.arange(len(asked))[:, None], asked].any()
    assert all(len(set(row)) == 4 for row in asked)
    Y = sim.Y.astype(float)
    loss_p, *_ = select.replay_curve(model, b, Y, asked, held)
    loss_o, *_ = select.replay_curve(model, b, Y, select.original_orders(avail, 4), held)
    assert loss_p[:, 4].mean() < loss_o[:, 4].mean()
