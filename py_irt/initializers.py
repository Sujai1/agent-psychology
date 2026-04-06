# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
A set of initializers to modify how IRT models are initialized.

For example, the expression disc * (skill - diff) permits two equivalent
solutions, one where disc/skill/diff are "normal", and another one where the
sign on each is flipped. Initializing difficulty helps push towards the intuitive
solution.
"""
import abc
import torch
import pyro
from rich.console import Console
from py_irt.dataset import Dataset, ItemAccuracy
import torch.nn.functional as F


console = Console()
INITIALIZERS = {}


def register(name: str):
    def decorator(class_):
        INITIALIZERS[name] = class_
        return class_

    return decorator


class IrtInitializer(abc.ABC):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def initialize(self) -> None:
        pass


@register("difficulty_sign")
class DifficultySignInitializer(IrtInitializer):
    def __init__(self, dataset: Dataset, magnitude: float = 3.0, n_to_init: int = 4):
        super().__init__(dataset)
        self._magnitude = magnitude
        self._n_to_init = n_to_init

    def initialize(self) -> None:
        """
        Initialize the hardest and easiest (by accuracy) n_to_init item difficulties.
        Set to magnitude.
        """
        item_accuracies = {}
        for item_ix, response in zip(self._dataset.observation_items, self._dataset.observations):
            if item_ix not in item_accuracies:
                item_accuracies[item_ix] = ItemAccuracy()

            item_accuracies[item_ix].correct += response
            item_accuracies[item_ix].total += 1

        sorted_item_accuracies = sorted(
            list(item_accuracies.items()), key=lambda kv: kv[1].accuracy
        )

        diff = pyro.param("loc_diff")
        for item_ix, accuracy in sorted_item_accuracies[: self._n_to_init]:
            item_id = self._dataset.ix_to_item_id[item_ix]
            console.log(f"Low Accuracy: {accuracy}, ix={item_ix} id={item_id}")
            diff.data[item_ix] = torch.tensor(
                self._magnitude, dtype=diff.data.dtype, device=diff.data.device
            )

        for item_ix, accuracy in sorted_item_accuracies[-self._n_to_init :]:
            item_id = self._dataset.ix_to_item_id[item_ix]
            console.log(f"High Accuracy: {accuracy}, ix={item_ix} id={item_id}")
            diff.data[item_ix] = torch.tensor(
                -self._magnitude, dtype=diff.data.dtype, device=diff.data.device
            )


@register("difficulty_from_accuracy")
class DifficultyFromAccuracyInitializer(IrtInitializer):
    def __init__(
        self,
        dataset: Dataset,
        *,
        eps: float = 1e-3,
        dims: int | None = None,
        jitter_std: float = 0.0,
        init_disc_std: float = 0.0,
        init_ability_std: float = 0.0,
    ):
        """
        Initialize item difficulties from empirical accuracies.

        For 1D 2PL with slope near 1 and ability mean near 0, a reasonable
        starting point is b_i ≈ -logit(accuracy_i). For MIRT (dims > 1), we
        broadcast the same scalar across all dimensions as a neutral start.

        Args:
            eps: clamp for accuracies to avoid infinities in logit
            dims: if provided, overrides inferred dimensionality (for MIRT)
        """
        super().__init__(dataset)
        self._eps = float(eps)
        self._dims = dims
        self._jitter_std = float(jitter_std)
        self._init_disc_std = float(init_disc_std)
        self._init_ability_std = float(init_ability_std)

    @staticmethod
    def _logit(p: torch.Tensor) -> torch.Tensor:
        return torch.log(p) - torch.log1p(-p)

    def initialize(self) -> None:
        # Compute empirical accuracy per item index
        counts: dict[int, ItemAccuracy] = {}
        for item_ix, response in zip(self._dataset.observation_items, self._dataset.observations):
            if item_ix not in counts:
                counts[item_ix] = ItemAccuracy()
            counts[item_ix].correct += response
            counts[item_ix].total += 1

        # Retrieve loc_diff to mutate in-place
        diff = pyro.param("loc_diff")  # [I] or [I, D]
        if diff.dim() == 1:
            D = 1
        else:
            D = diff.size(1)
        if self._dims is not None:
            D = int(self._dims)

        for item_ix, acc in counts.items():
            # Clamp accuracy into (eps, 1-eps)
            if acc.total <= 0:
                continue
            p = max(self._eps, min(1.0 - self._eps, acc.accuracy))
            b0 = -self._logit(torch.tensor(p, dtype=diff.dtype, device=diff.device))
            if D == 1:
                diff.data[item_ix] = b0
            else:
                # Broadcast base difficulty then add small per-dim jitter to break symmetry
                diff.data[item_ix, :] = b0
                if self._jitter_std > 0:
                    diff.data[item_ix, :] += torch.randn(D, device=diff.device, dtype=diff.dtype) * self._jitter_std
            item_id = self._dataset.ix_to_item_id[item_ix]
            console.log(f"Init diff from acc: id={item_id} acc={acc.accuracy:.3f} b0={float(b0):.3f}")

        # Optionally break symmetry further by initializing discrimination and abilities
        if self._init_disc_std > 0:
            try:
                disc = pyro.param("loc_disc")  # [I, D]
                disc.data += torch.randn_like(disc) * self._init_disc_std
                console.log(f"Initialized disc with std={self._init_disc_std}")
            except KeyError:
                pass


@register("mirt_pca")
class MirtPCAInitializer(IrtInitializer):
    def __init__(
        self,
        dataset: Dataset,
        *,
        dims: int,
        disc_scale: float = 0.5,
        ability_scale: float = 0.5,
        center: str = "item",  # "item" or "global"
        init_ability_std: float = 0.0,
    ):
        """
        PCA/SVD-based warm start for MIRT.

        Builds an item-by-subject response matrix X (0/1), mean-centers rows
        (items) by default, computes SVD X = U Σ V^T, and initializes:
          loc_disc    ← disc_scale * U[:, :D]
          loc_ability ← ability_scale * V[:, :D] (optionally times Σ)

        Args:
            dims: number of latent dimensions (D)
            disc_scale: scaling applied to U loadings
            ability_scale: scaling applied to V scores (Σ can be large)
            center: 'item' to subtract item means; 'global' to subtract global mean
        """
        super().__init__(dataset)
        self._dims = int(dims)
        self._disc_scale = float(disc_scale)
        self._ability_scale = float(ability_scale)
        self._center = center
        self._init_ability_std = float(init_ability_std)

    def initialize(self) -> None:
        # Access shapes and devices from params
        try:
            disc = pyro.param("loc_disc")  # [I, D]
            diff = pyro.param("loc_diff")  # [I, D] or [I]
            ability = pyro.param("loc_ability")  # [S, D]
        except KeyError:
            console.log("PCA init: required parameters not found; skipping")
            return

        I = disc.size(0)
        D = min(self._dims, disc.size(1))
        S = ability.size(0)
        device = disc.device
        dtype = disc.dtype

        # Build item x subject response matrix X
        X = torch.empty((I, S), dtype=dtype, device=device)
        X.fill_(float('nan'))
        for subj_ix, item_ix, resp in zip(
            self._dataset.observation_subjects,
            self._dataset.observation_items,
            self._dataset.observations,
        ):
            # Assumes one observation per pair; last write wins otherwise
            X[item_ix, subj_ix] = float(resp)

        # Replace any NaNs with row means or zeros if entire row missing
        row_means = torch.nanmean(X, dim=1, keepdim=True)
        # Where an entire row is NaN, set mean to 0
        row_means = torch.nan_to_num(row_means, nan=0.0)
        X = torch.where(torch.isnan(X), row_means, X)

        # Center
        if self._center == "item":
            Xc = X - X.mean(dim=1, keepdim=True)
        else:  # global
            Xc = X - X.mean()

        # SVD
        try:
            U, Svals, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            console.log("PCA init: SVD failed; skipping")
            return

        # Initialize disc and abilities from first D components
        disc.data[:, :D] = self._disc_scale * U[:, :D]
        ability.data[:, :D] = self._ability_scale * Vh[:D, :].T
        console.log(
            f"Initialized MIRT via PCA: dims={D}, disc_scale={self._disc_scale}, ability_scale={self._ability_scale}"
        )

        if self._init_ability_std > 0:
            try:
                ability = pyro.param("loc_ability")  # [S] or [S, D]
                ability.data += torch.randn_like(ability) * self._init_ability_std
                console.log(f"Initialized abilities with std={self._init_ability_std}")
            except KeyError:
                pass
