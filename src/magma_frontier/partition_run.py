"""Assemble the confound audit, the representation and the partition attack into one run."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from magma_frontier.attacks.partition import PartitionResult, partition_attack
from magma_frontier.embed.representation import fit_transform
from magma_frontier.eval.confound import ConfoundReport, audit


@dataclass(frozen=True, slots=True)
class PartitionRun:
    confound: ConfoundReport
    partition: PartitionResult
    ari_heldout: float
    n_heldout: int
    n_components: int
    explained_variance: float
    n: int
    n_tenants: int


def run_partition(fs, n_components: int = 128, seed: int = 0) -> PartitionRun:
    """Audit confounds, fit the representation on a task-grouped fold, then partition."""
    confound = audit(fs)

    groups = np.asarray(fs.task_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, test_idx = next(splitter.split(fs.X, np.asarray(fs.tenant_ids), groups))

    rep = fit_transform(fs, train_idx, n_components=n_components, seed=seed)

    # The representation is fitted on a fold, but the attack scores the whole pool: a real
    # buyer holds every row of the release, they simply hold no labels.
    result = partition_attack(rep.Z, np.asarray(fs.tenant_ids), seed=seed)

    # Most rows the whole-pool score covers were in-sample for the SVD basis. Score the
    # held-out tasks alone as well: if the two agree, the whole-pool number is honest; if
    # the held-out one is markedly lower, the whole-pool number is optimistic and the
    # caveat has to travel with the headline.
    heldout = partition_attack(
        rep.Z[test_idx], np.asarray(fs.tenant_ids)[test_idx], seed=seed
    )

    return PartitionRun(
        confound=confound,
        partition=result,
        ari_heldout=heldout.ari,
        n_heldout=int(test_idx.size),
        n_components=rep.n_components,
        explained_variance=rep.explained_variance,
        n=int(fs.X.shape[0]),
        n_tenants=int(np.unique(fs.tenant_ids).size),
    )
