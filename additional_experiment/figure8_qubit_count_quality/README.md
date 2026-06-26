# Figure 8 Qubit Count Revision

This folder reconstructs the MDKP/MIS qubit-count versus recovered hardware
solution-quality figure from the checked-in hardware `main_table` CSV files.

The only requested label change is applied in code:

```python
ax.set_xlabel("Qubit Count")
```

## Files

- `plot_figure8_qubit_count_quality.py`: plotting script.
- `figure8_qubit_count_quality_data.csv`: exact data used by the plot.
- `fig8_qubit_count_vs_quality_revised.pdf`: regenerated figure.
- `fig8_qubit_count_vs_quality_revised.png`: raster preview.

## Rerun

```bash
.venv/bin/python additional_experiment/figure8_qubit_count_quality/plot_figure8_qubit_count_quality.py
```
