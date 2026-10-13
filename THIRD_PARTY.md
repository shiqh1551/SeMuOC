# Third-party components

- `ts_datasets/` contains dataset loaders derived from Salesforce Merlion. Existing copyright and BSD-3-Clause headers are retained. Upstream: https://github.com/salesforce/Merlion .
- `affiliation/` contains affiliation-based anomaly evaluation utilities. Upstream: https://github.com/ahstat/affiliation-metrics-py .
- RPA metrics use Salesforce Merlion; VUS metrics use the `vus` package.
- The frozen text encoder is `intfloat/e5-base-v2`: https://huggingface.co/intfloat/e5-base-v2 . Model weights are not included.

The bundled source inherits existing components from the local time-series anomaly detection codebase. No new blanket license is assigned by this export. Existing attribution and license notices must be preserved.
