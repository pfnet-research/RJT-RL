# Penalized LogP optimization
This example explore molecules optimizing the penalized LogP function as a reward.

## P1
- Conditions: step reward, no dup penalty
- Command:
```
bash examples/penalized_logp/run_p1_step_nodup.sh
```
- Output: `<topdir>/results/p1`
- Analysis: [analysis.ipynb](analysis.ipynb)

## P2
- Conditions: final reward, dup penalty
- Command:
```
bash examples/penalized_logp/run_p2_final_dup.sh
```
- Output: `<topdir>/results/p2`
- Analysis: [analysis.ipynb](analysis.ipynb)

## P3
- Conditions: step reward, dup penalty
- Command:
```
bash examples/penalized_logp/run_p3_step_dup.sh
```
- Output: `<topdir>/results/p3`
- Analysis: [analysis.ipynb](analysis.ipynb)
