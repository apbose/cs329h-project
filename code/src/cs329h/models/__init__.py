"""Choice models for binary choices between two lotteries.

Every model gives the probability that person ``i`` chooses the riskier option of question
``q``, ``eps / 2 + (1 - eps) * sigmoid(s_iq)``, and a population prior over the person
parameters whose mean depends on the 0--10 self-rating when it is used. See
:mod:`cs329h.models.base` for the interface that selection and evaluation use.

* M1 :class:`ExpectedUtility` -- expected utility with constant absolute risk aversion (grid).
* M2 :class:`ProspectTheory` -- cumulative prospect theory (shared importance samples).
* M3 :class:`FactorModel` -- low-rank factor model (Laplace).
* M4 :class:`NeuralModel` -- neural network with a person embedding (variational fit,
  importance-sampled posterior), optionally pretrained on synthetic M2 choices.
"""

from .base import SupportModel, entropy, lapse_prob, normalize, person_loglik, rating_z
from .cpt import CPTParams, ProspectTheory, cpt_value
from .eu import EUParams, ExpectedUtility, cara_utility
from .factor import FactorModel
from .latent import LaplaceModel, LatentParams
from .neural import NeuralModel

__all__ = [
    "CPTParams", "EUParams", "ExpectedUtility", "FactorModel", "LaplaceModel", "LatentParams", "MODEL_CLASSES",
    "NeuralModel", "ProspectTheory", "SupportModel", "cara_utility", "cpt_value", "entropy", "lapse_prob",
    "make_model", "normalize", "person_loglik", "rating_z",
]

MODEL_CLASSES = ("M1", "M2", "M3", "M4", "M4-pre")


def make_model(name: str, use_rating: bool = True, **kwargs):
    """An unfitted model of class ``name`` (one of :data:`MODEL_CLASSES`)."""
    if name == "M1":
        return ExpectedUtility(use_rating=use_rating, **kwargs)
    if name == "M2":
        return ProspectTheory(use_rating=use_rating, **kwargs)
    if name == "M3":
        return FactorModel(use_rating=use_rating, **kwargs)
    if name == "M4":
        return NeuralModel(use_rating=use_rating, **kwargs)
    if name == "M4-pre":
        return NeuralModel(use_rating=use_rating, pretrain=True, **kwargs)
    raise ValueError(f"unknown model class {name!r}")
