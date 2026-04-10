# Baseline Models Comparison

This project compares three Gemma-family models on GLOBEM-style tasks using one notebook per model.

## Models

- Gemma 3
- MedGemma 1.5
- Gemma 4 backbone

## Dataset

Place your CSV at `data/globem/globem_eval.csv` with these required columns:

- `sample_id`
- `image_path_or_url`
- `question`
- `gold_answer`

## Files

- `config/eval_config.json` - shared config (model ids, prompt, generation settings)
- `notebooks/eval_gemma3_globem.ipynb`
- `notebooks/eval_medgemma15_globem.ipynb`
- `notebooks/eval_gemma4_backbone_globem.ipynb`
- `results/` - run artifacts from each notebook

## Result naming

Each notebook writes two artifacts:

- `globem_{model_slug}_{system_prompt_id}_{yyyymmdd_hhmmss}_predictions.csv`
- `globem_{model_slug}_{system_prompt_id}_{yyyymmdd_hhmmss}_metrics.json`

## Quick run

1. Open one notebook.
2. Confirm `MODEL_KEY` in the first config cell.
3. Run all cells.
4. Inspect files written in `results/`.
