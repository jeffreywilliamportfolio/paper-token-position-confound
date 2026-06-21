# When Routing Entropy Tracks Length, Not Complexity

A Cross-Model Token-Position Confound in MoE Interpretability — paper source, figures, and reproducibility materials.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20779602.svg)](https://doi.org/10.5281/zenodo.20779602)

**Author:** Jeffrey W. Shorthill (independent researcher) · `jws299792@icloud.com`
**Version:** 1.0 (June 2026) · preprint, not peer reviewed
**DOI:** [10.5281/zenodo.20779602](https://doi.org/10.5281/zenodo.20779602) (concept DOI — resolves to the latest version)
**License:** [CC BY 4.0](LICENSE)

## What this is

Routing entropy — the Shannon entropy of a mixture-of-experts (MoE) router's per-token
weight distribution — is often read as a sign of how hard a model is working. This paper
shows that, averaged over prefill tokens, that signal is confounded by prompt **token
count** and **token position**: later prefill positions carry higher routing entropy
(they attend to more context under causal attention), so longer prompts lift the
all-token mean. Measured at the final prefill token, the apparent complexity gradient
disappears and the extreme levels reverse. Demonstrated on DeepSeek V3.1 and
Qwen3.5-397B, with a DeepSeek R1 replication.

## Repository layout

| Path | Contents |
|---|---|
| `main.tex` | Paper source (LaTeX). |
| `refs.bib` | Bibliography (every entry verified against an authoritative record). |
| `figures/` | The three figures, as `.pdf` and `.png`. |
| `make_figures.py` | Regenerates the figures and reported correlations from the source values. Runs no model. |
| `SOURCES.md` | Claim-by-claim source-to-value index. |
| `main.pdf` | Built PDF of the paper. |

## Build

```bash
latexmk -pdf main.tex      # produces main.pdf
python make_figures.py     # regenerates figures/ (no model is run)
```

## Data availability

The figures and correlations are recomputed from a preserved capture archive
(`token-confound-archive/`, referenced in `SOURCES.md`); this repository carries the
paper, figures, and the script that reproduces them from the archived per-prompt
records. Raw activation/router captures are retained separately and are available from
the author for verification.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the preprint (version 1.0, 2026).

## AI-use disclosure

Generative AI (Anthropic's Claude) was used for drafting, organization, and
bibliography formatting. The author verified every reported value and reference and
takes full responsibility for the content.
