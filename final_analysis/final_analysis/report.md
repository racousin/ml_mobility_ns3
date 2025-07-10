# Trajectory Generation Experiment Analysis Report

Generated on: 2025-07-10 13:35:51

## Summary

Total experiments analyzed: 7

## Best Experiments by Metric

### Best Reconstruction Loss
| experiment        |   recon_total_loss |   recon_speed_mae |
|:------------------|-------------------:|------------------:|
| optimal_medium_v2 |           0.626439 |         0.0168872 |
| optimal_medium_v1 |           0.873413 |         0.0176444 |
| optimal_medium_v3 |           0.959806 |         0.0167136 |
| optimal_small_v1  |           1.50103  |         0.0164568 |
| optimal_small_v3  |           1.50822  |         0.0170021 |

### Best Speed MAE (Reconstruction)
| experiment        |   recon_speed_mae |   recon_total_loss |
|:------------------|------------------:|-------------------:|
| optimal_small_v1  |         0.0164568 |           1.50103  |
| optimal_medium_v3 |         0.0167136 |           0.959806 |
| optimal_small_v4  |         0.0168788 |           1.53255  |
| optimal_medium_v2 |         0.0168872 |           0.626439 |
| optimal_small_v3  |         0.0170021 |           1.50822  |

### Best Generation Quality (Speed)
| experiment        |   gen_speed_mae_avg |   gen_total_distance_mae_avg |
|:------------------|--------------------:|-----------------------------:|
| optimal_small_v2  |         0.000953312 |                      5.58824 |
| optimal_medium_v1 |         0.00125228  |                      3.97239 |
| optimal_medium_v2 |         0.00125948  |                      3.86497 |
| optimal_small_v3  |         0.00132736  |                      3.9336  |
| optimal_small_v1  |         0.00153821  |                      3.59418 |

## Configuration Analysis

### Latent Dimension Impact
|   latent_dim |     mean |        std |   count |
|-------------:|---------:|-----------:|--------:|
|           16 | 1.73594  |   0.379911 |       3 |
|           20 | 1.50822  | nan        |       1 |
|           28 | 0.626439 | nan        |       1 |
|           32 | 0.916609 |   0.061089 |       2 |

### Hidden Dimension Impact
|   hidden_dim |     mean |        std |   count |
|-------------:|---------:|-----------:|--------:|
|          128 | 1.72783  |   0.386622 |       3 |
|          144 | 1.53255  | nan        |       1 |
|          240 | 0.959806 | nan        |       1 |
|          256 | 0.749926 |   0.174637 |       2 |

### Beta Parameter Impact
|   beta |     mean |         std |   count |
|-------:|---------:|------------:|--------:|
| 0.0005 | 2.17424  | nan         |       1 |
| 0.001  | 1.51393  |   0.0165176 |       3 |
| 0.002  | 0.916609 |   0.061089  |       2 |
| 0.003  | 0.626439 | nan         |       1 |

## Files Generated

- `detailed_analysis.json`: Complete analysis results
- `experiment_summary.csv`: Summary table of all experiments
- `training_curves.png`: Training curves comparison
- `generation_comparison.png`: Generation quality comparison

## Recommendations

Based on the analysis:
1. Best overall model: optimal_medium_v2
2. Best for speed accuracy: optimal_small_v1
3. Most balanced performance: optimal_medium_v2
