"""Is the tenant signal actually tenant signal, or is it task composition?

arXiv:2402.07841 found that an apparent membership-inference signal was really temporal
shift: the attack detected a nuisance covariate correlated with the label rather than the
thing it claimed to detect. The analogous trap here is task composition. Group-splitting
stops a task spanning train and test, but it cannot stop tenants from attempting different
task sets in the first place — and attrition that varies by tenant is exactly how that
happens.

So: measure how well the task label alone predicts the tenant, with no trace features at
all. If that beats the majority baseline by much, task composition carries tenant
information and every separability number needs the caveat attached.

This is computed directly rather than estimated with a classifier. Integer-coding an
arbitrary categorical and asking a tree to find interval splits makes the answer depend on
label sort order and cross-validation fold boundaries rather than on the data — and the
question here is whether the information is present at all, not whether a model can
generalize to unseen tasks.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfoundReport:
    tenant_from_task_accuracy: float
    task_chance: float
    median_sessions_per_task: float
    min_task_jaccard: float
    mean_task_jaccard: float
    tenant_task_counts: dict[str, int]


def _task_sets(tenant_ids: np.ndarray, task_ids: np.ndarray) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for tenant, task in zip(tenant_ids, task_ids):
        sets.setdefault(str(tenant), set()).add(str(task))
    return sets


def audit(fs) -> ConfoundReport:
    """Report how much tenant information task composition alone carries."""
    tenant_ids = np.asarray(fs.tenant_ids)
    task_ids = np.asarray(fs.task_ids)

    tenants = np.unique(tenant_ids)
    if tenants.size < 2:
        raise ValueError(f"confound audit needs at least two tenants, got {tenants.size}")

    # How well does the task label alone predict the tenant? Computed directly rather
    # than estimated with a model: the question is information-theoretic (does task
    # composition CARRY tenant information) not predictive, and integer-coding an
    # arbitrary categorical for a tree makes the answer depend on label sort order and
    # on cross-validation fold boundaries rather than on the data.
    #
    # The best possible rule is "given this task, guess its most common tenant". Its
    # accuracy is the share of rows that rule gets right.
    by_task: dict[str, Counter] = {}
    for tenant, task in zip(tenant_ids, task_ids):
        by_task.setdefault(str(task), Counter())[str(tenant)] += 1
    correct = sum(counts.most_common(1)[0][1] for counts in by_task.values())
    accuracy = float(correct) / float(len(tenant_ids))

    counts = Counter(tenant_ids)
    chance = counts.most_common(1)[0][1] / len(tenant_ids)

    sets = _task_sets(tenant_ids, task_ids)
    jaccards = []
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = sets[a] | sets[b]
            jaccards.append(len(sets[a] & sets[b]) / len(union) if union else 1.0)

    # Accuracy alone is ambiguous: with one session per task the best-guess rule hits
    # 1.0 by memorization, which is indistinguishable from a real confound. Report how
    # many sessions back each task so a reader can tell those two apart.
    return ConfoundReport(
        tenant_from_task_accuracy=accuracy,
        task_chance=float(chance),
        median_sessions_per_task=float(
            np.median([sum(counts.values()) for counts in by_task.values()])
        ),
        min_task_jaccard=float(min(jaccards)) if jaccards else 1.0,
        mean_task_jaccard=float(np.mean(jaccards)) if jaccards else 1.0,
        tenant_task_counts={name: len(tasks) for name, tasks in sets.items()},
    )
