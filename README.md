# magma-frontier

Can you identify the company that contributed a set of AI agent traces, from the traces
alone, after every byte of text is stripped out?

Mostly yes if you already have labelled examples of their work. **No if all you have is the
pool.** And the contributors whose data is worth most are the *least* identifiable, not the
most.

## Results

| Attack | What the adversary has | Result |
|---|---|---|
| Closed-set re-identification | Labelled traces per contributor | **4.0x** the majority baseline, TPR 0.087 at 1% FPR |
| Unsupervised partition | Only the pooled release | **ARI 0.020** — no recovery |
| Value vs exposure | Both | **rho -0.569**, 95% CI [-0.774, -0.202] |

Corpus: 6878 agent sessions from 22 distinct models solving the same ~108 tasks against the
same tool environments ([Toolathlon](https://huggingface.co/datasets/hkust-nlp/Toolathlon-Trajectories)).
All text removed. Session timing removed. What remains is sequence structure: which tool
families were called in what order, retry and error patterns, argument shapes.

Full write-ups in [`docs/results/`](docs/results/).

## What is and is not new here

**Not new.** That agent traces carry structural fingerprints is established — prior work
identifies the *model* or the *agent framework* at 90-98% F1 from UI traces
([arXiv:2605.14786](https://arxiv.org/abs/2605.14786)), encrypted network traffic
([arXiv:2510.07176](https://arxiv.org/abs/2510.07176)), terminal commands
([arXiv:2605.01186](https://arxiv.org/abs/2605.01186)) and commit structure
([arXiv:2601.17406](https://arxiv.org/html/2601.17406v1)). That rare records are more
re-identifiable is older still (Sweeney 2002; Narayanan & Shmatikov 2008).

**New.** The identified entity is the *contributor*, not the model or framework. The
setting is a pooled multi-contributor release evaluated as one. And the value-risk
relationship is measured at contributor level, where the literature disagrees with itself:
strongly positive at record level (Wen, Backes & Zhang, NDSS 2025), absent at
federated-client level (El Mestari et al., SECRYPT 2025). This lands with the latter.

## Reproducing

```bash
uv sync
uv run python -m magma_frontier.cli download --dest data/toolathlon   # gated, needs HF login
uv run python -m magma_frontier.cli smoke     --data data/toolathlon --seeds 5 --exclude-timing
uv run python -m magma_frontier.cli partition --data data/toolathlon --exclude-timing
uv run python -u scripts/frontier_analysis.py cache               # ~42 min, once
uv run python -u scripts/frontier_analysis.py report              # minutes, repeatable
```

`out/frontier_cache.npz` (6.8 MB) holds everything the frontier numbers are derived from,
so `report` reproduces them without the 1.9 GB corpus.

## How it is built

`features/` is the only place text is dropped, so the "no text survives" claim is one file
a reader can check in a minute rather than a promise spread across the codebase. Contributor
labels travel beside the feature matrix, never inside it. Every fitted object — vocabulary
weights, scaler, SVD basis — is fitted on the training fold alone.

Attacks are scored the way Carlini et al. ([arXiv:2112.03570](https://arxiv.org/abs/2112.03570))
argue privacy attacks must be: TPR at a fixed low false-positive rate against a **measured**
null, not average-case accuracy against a nominal one. Every headline carries a control that
would expose it if the pipeline were broken — shuffled labels for the supervised attack, a
task-composition audit for the partition, a majority-baseline and AUC check for the value
estimator.

## Things that are wrong with it

Stated here because they are stated in the write-ups too.

- **One of the two value estimators does not work.** Its model scores below the majority
  baseline (0.729 against 0.762, AUC 0.471), so its per-contributor numbers are differences
  between two unskilled models. Reported as a failed measurement, not as a null.
- **The coverage axis reproduces itself at only 0.466**, so a single-seed ranking is about
  half noise. The headline uses a 10-seed mean and a bootstrap interval for that reason.
- **n = 22.** Only large effects are detectable. The confidence interval is wide.
- **The contributor here is a model, not a company.** Real companies differ in more
  dimensions, so separability is a lower bound and the protective reading of the null is an
  upper bound.
- **Coverage value correlates +0.589 with task success rate.** Anyone pricing contributors
  on coverage would substantially be paying them for succeeding.
- Depth-2 taxonomy is computationally impractical; depth-0 needs the namespace table
  extended from 23 to 98 entries first. Both unrun.

## Licence

Code under MIT. The corpus is third-party, CC-BY-4.0, and is not redistributed here.
