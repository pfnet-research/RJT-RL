# Similarity optimization
This example explore molecules similar to the specified chemical structure.
Vortioxetine and celecoxib are used as examples.

## Vortioxetine rediscovery experiments
### S1
- Conditions: step reward, no dup penalty
- Command:
```
bash examples/similarity/run_s1_step_nodup.sh
```
- Output: `<topdir>/results/s1`
- Analysis: [analysis.ipynb](analysis.ipynb)

### S2
- Conditions: final reward, dup penalty
- Command:
```
bash examples/similarity/run_s2_final_dup.sh
```
- Output: `<topdir>/results/s2`
- Analysis: [analysis.ipynb](analysis.ipynb)

### S3
- Conditions: step reward, dup penalty
- Command:
```
bash examples/similarity/run_s3_step_dup.sh
```
- Output: `<topdir>/results/s3`
- Analysis: [analysis.ipynb](analysis.ipynb)

## Celecoxib rediscovery experiments
### C1
- Conditions: step reward, no dup penalty
- Command:
```
bash examples/similarity/run_c1_step_nodup.sh
```
- Output: `<topdir>/results/c1`
- Analysis: [analysis.ipynb](analysis.ipynb)

### C2
- Conditions: final reward, dup penalty
- Command:
```
bash examples/similarity/run_c2_final_dup.sh
```
- Output: `<topdir>/results/c2`
- Analysis: [analysis.ipynb](analysis.ipynb)

### C3
- Conditions: step reward, dup penalty
- Command:
```
bash examples/similarity/run_c3_step_dup.sh
```
- Output: `<topdir>/results/c3`
- Analysis: [analysis.ipynb](analysis.ipynb)
