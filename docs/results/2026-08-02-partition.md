# Phase 1B-1 partition result — 2026-08-02

**Pooling resists unsupervised partition.** A buyer holding a pooled trace release with no
labels cannot recover the contributor structure. ARI 0.020 at seed 0, range 0.0156-0.0201
over five seeds, against a chance-corrected baseline of 0.0.

This is the negative outcome, and it was committed to in advance as being worth reporting
as prominently as the positive one. It is also the more commercially useful of the two: the
attack that would actually harm a marketplace is harder than the one the literature runs.

## Run

```
uv run python -m magma_frontier.cli partition --data data/toolathlon --depth 1 \
    --seed 0 --components 128 --exclude-timing

uv run python -m magma_frontier.cli partition --data data/toolathlon --depth 1 \
    --seed 0 --components 128
```

The first command produces the structure-only column, the second the with-timing column.
Commit `bfaabe6`. Corpus `hkust-nlp/Toolathlon-Trajectories`, all 66 JSONL files,
6878 sessions across 22 tenants.

## Result

| | Structure only | With timing |
|---|---|---|
| **ARI (whole pool)** | **0.020** | 0.023 |
| ARI (held-out tasks only) | 0.017 | 0.020 |
| AMI | 0.069 | 0.081 |
| Clusters | 22 | 22 |
| Largest cluster share | 0.162 | 0.180 |
| Representation | 128 components, 0.996 explained variance | same |

Per-tenant purity, structure only: 0.365 (deepseek-3.2-thinking) down to 0.147 (gpt-5),
against a floor of ~0.162 set by the largest cluster's share alone.

## The confound audit: task composition is not smuggling tenant information in

```
tenant predicted from task alone:  0.047107   (majority baseline 0.046961)
task-set jaccard:                  min 0.935, mean 0.979
median sessions per task:          65
```

Task identity gives **+0.0001** lift over the majority baseline, which is the floor set by
the balanced design. In absolute terms that is 0.047107 against 0.046961, a difference of
about one session in 6878. Tenants attempted 103 to 108 of the 108 tasks, so the task sets
are near-identical by construction. And 65 sessions back each task, so the accuracy figure
is not a small-group artifact, the case where a near-1.0 score would mean memorization
rather than structure.

What this rules out is a specific confound: that the clustering could have recovered tenants
by proxy, through tenants having run different work. It could not, because they ran the same
work. That is necessary for interpreting the null, but it is not an argument that the
experiment had the power to detect anything. That argument is below.

## Why this is a null and not a failure to measure

A null result is only interesting if the experiment could have detected something. Four
checks establish that it could.

**The tenant signal survives into the representation.** A supervised HistGB trained on `Z`
alone, same task-grouped split, seed 0, reaches **0.163** accuracy against the 0.047
majority baseline, 3.46x. The same classifier on the raw features `X` under identical
conditions reaches **0.177**, 3.77x. `Z` retains about **92%** of the raw-feature accuracy.
The signal is sitting in the 128 components. KMeans simply cannot convert it into a
partition.

**The discriminative directions are the high-variance ones.** Component 0 alone carries
η²=0.210 and 21.5% of `Z`'s variance. The variance-weighted mean η² across components is
**0.084** against **0.013** unweighted, so tenant-discriminative directions are
systematically the ones with the most variance. SVD truncation is not discarding the signal.

**The clustering does recover latent structure, just not the contributor.** Scored against
task labels instead of tenant labels, the same k=22 clustering gives AMI **0.276**
(ARI 0.041), rising to **0.387** at k=108, the true task count. Against tenant labels it
gives AMI 0.069. The clusters are tracking what the agent was asked to do, not who ran it.

**The result survives every alternative geometry tried.** All of them are within ARI
[0.001, 0.038]:

| Variant | ARI |
|---|---|
| Headline: `Z`, k=22 | 0.020 |
| L2-normalised `Z` | 0.027 |
| Whitened `Z` | 0.001 |
| Block-rebalanced | 0.008 |
| n-gram block only | 0.003 |
| Scalar block only | 0.020 |
| Raw dense features, no SVD | 0.020 |
| k ∈ {5, 11, 44, 100} | 0.010-0.022 |
| KMeans on LDA(`Z`), fitted with the labels | 0.038 |
| KMeans on the 20 most tenant-discriminative components, selected with the labels | 0.008 |

The last two rows are label-cheating upper bounds and they fail too. An adversary handed the
answer key for which subspace to look in still cannot carve up the pool.

## Why ARI rather than purity

Per-tenant purity ranges 0.147 to 0.365, which looks like recovered structure until the
cluster size distribution is accounted for. `largest_cluster_share` is 0.162, so a tenant
whose rows scatter proportionally already scores about 0.162 for nothing. Adjusted Rand
Index corrects for chance agreement and for the cluster size distribution, which is why it
lands at 0.020 where purity suggests otherwise.

Choosing a chance-corrected metric before seeing the number is the reason this reads as a
null rather than as a weak positive.

## The held-out cross-check

`run_partition` fits the representation on a task-grouped 70% training fold but scores the
whole pool, because a real buyer holds every row of a release they purchased. That means
most scored rows were in-sample for the SVD basis, so the whole-pool figure could in
principle be optimistic.

It is not. Over five seeds, held-out ARI ranges 0.0115 to 0.0308 across 2108 sessions,
straddling the whole-pool range of 0.0156 to 0.0201. At seed 0 the two happen to sit close
together (0.017 against 0.020), but that is a coincidence of the seed: at seed 2 held-out is
1.7x whole-pool and at seed 3 it is 0.6x. What the comparison supports is the weaker and
correct claim that there is no evidence the whole-pool number is optimistic. It does not
support the claim that the two agree.

## Timing without labels

In Phase 1A's supervised attack, three session-timing columns carried **41%** of the
accuracy above baseline. Here they move ARI from 0.020 to 0.023, which looks like nothing.
It is not nothing, and it is not a smaller share than it was in Phase 1A.

Over five seeds, structure-only ARI is **0.0186** (0.0156-0.0201) and with-timing ARI is
**0.0249** (0.0217-0.0273). The two ranges do not overlap: every with-timing seed exceeds
every structure-only seed. Timing accounts for **25%** of the with-timing ARI, comparable to
the 41% it carried in the supervised attack. It carries a similar share of a much smaller
number.

Timing is in fact the single most cluster-informative signal in the corpus. KMeans on the
three timing columns alone, standardised, k=22, no labels, gives ARI **0.034-0.036** and AMI
**0.128-0.130** over five seeds. That is 1.7-1.9x the headline ARI of 0.020 and 1.9x the
headline AMI of 0.069. `duration_s` on its own gives ARI 0.028. Median session duration
spans 31s (`gemini-2.5-flash`) to 491s (`kimi-k2-0905`), a 16x spread. Timing looks
negligible in the headline only because it is diluted to 3 of the 20 standardised scalar
columns inside a 128-dimension SVD.

So an adversary does not need labels to exploit timing. Clustering on timing alone beats
clustering on the full representation. The point is that it does not get anyone to a usable
partition either way: 0.035 is a failed attack for the same reason 0.020 is. The unsupervised
attack fails with timing and without it, and the reason to keep timing out of the headline is
the Phase 1A reason, that it measures serving-endpoint speed rather than contributor
behaviour, not that it is inert.

## What this means alongside Phase 1A

| Attack | Adversary knowledge | Result |
|---|---|---|
| Closed-set re-identification | Labelled examples per tenant | 4.0x majority baseline, TPR 0.087 @ 1% FPR |
| Unsupervised partition | Nothing but the release | ARI 0.020, no recovery |

The gap between these is the finding. Tenant signal is present in trace structure. Phase 1A
established that under a clean shuffled-label control, and the supervised probe on `Z` above
confirms it survives the representation used here. But converting that signal into a
partition of an unlabelled pool does not follow from it.

For a marketplace, the practical reading: a buyer who already holds labelled traces from a
known supplier can recognise more of that supplier's traces in a pool. A buyer holding only
the pool cannot carve it up by contributor. Those are different threats with different
mitigations, and only the first is demonstrated here.

## Limits

- **The headline is a single seed.** The project's own Phase 1A standard is a five-seed
  range and this headline does not meet it. Over five seeds the whole-pool structure-only
  ARI is 0.0186, range 0.0156-0.0201. That interval is the honest figure, not 0.020 read to
  three significant figures.
- **One corpus, one clustering algorithm.** Every variant in the robustness table is
  KMeans. Setting k to the true tenant count is a strong assumption in the adversary's
  favour and it still fails. A different algorithm (HDBSCAN, spectral) might do better; this
  is evidence about this attack, not a proof that no unsupervised attack works.
- **0.996 explained variance is not the reassurance it looks like.** 94.6% of the total
  variance is the 17 standardised scalar columns (block variance 16.02) against 0.92 for all
  2643 TF-IDF n-gram columns combined, because `TfidfTransformer` L2-normalises each row, so
  a session's entire n-gram profile has norm exactly 1.0 against 3.25 for the scalar block.
  Retaining 0.996 of the variance is therefore close to automatic and mostly reports that the
  17 scalars were kept. KMeans on `Z` is about 95% driven by them: 95.7% of `Z`'s variance
  sits in the 15 components with more than 50% scalar mass, and clustering the n-gram block
  alone gives ARI 0.003. This does not change the conclusion, but the published number tells
  a reader something other than what they would assume it does.
- **The n-gram vocabulary is still fitted over the whole corpus.** `extract()` takes a
  `vocabulary=` argument and `FeatureSet` exposes `ngram_vocabulary`, but nothing in
  `build_corpus`, `smoke.py` or `partition_run.py` passes or reads them, so the plan's global
  constraint that anything fitted is fitted on the training fold only is not met by the run
  that produced this headline. The IDF weighting, the scaler and the SVD basis are fitted on
  the training fold; the column set is not. The effect is benign, because a column with zero
  training support receives zero loading in every SVD component, but Phase 1A disclosed its
  transduction and this run inherits it.
- **The tenant here is a model, not a company.** Real companies differ in more dimensions,
  so this is a lower bound on separability and therefore an *upper* bound on how protective
  pooling is. Pooling may protect less in the field than it does here.
- **Depth 1 only.** Depth 2 remains computationally impractical and depth 0 needs the
  taxonomy table extended from 23 to 98 namespaces first.
- **Not run:** membership inference and attribute inference, which are Phase 1B-2 and are
  the attacks most likely to succeed where partition failed.

## Carried into Phase 1B-2

The frontier's shape now depends on this result. Per-tenant exposure cannot come from
partition purity, because partition does not work. It has to come from the supervised
attacks instead, which changes how the value-risk join is built.
