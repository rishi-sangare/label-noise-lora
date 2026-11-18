| Training data | Runs | Accuracy (mean ± sd) | Macro-F1 | Valid format |
|---|---|---|---|---|
| No fine-tune (0.5B) | 1 | 54.2% | 45.4 | 98.8% |
| Clean labels | 3 | 87.9% ± 2.4 | 87.7 | 100.0% |
| 20% random noise | 3 | 85.1% ± 1.2 | 84.8 | 99.9% |
| 40% random noise | 3 | 81.1% ± 4.2 | 80.5 | 100.0% |
| 20% systematic noise | 3 | 81.7% ± 5.3 | 81.3 | 100.0% |
| 40% noise, judge-filtered | 3 | 79.3% ± 0.4 | 77.3 | 100.0% |
| Clean, same size as filtered | 3 | 87.5% ± 2.8 | 87.0 | 100.0% |
| 3B judge, no fine-tune | 1 | 69.2% | 67.8 | 93.6% |
