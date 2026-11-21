# Research (shelved)

This package holds the original lab-measurement workflow: importing field and laboratory
observations, matching them to Sentinel-2 scenes, human review, versioned training exports,
and the NLA 2017 chlorophyll preprocessing. It is **not** part of the deployed product, which
estimates water-quality parameters from multispectral imagery using a model trained on public
reflectance data.

It depends on `aquadoc_core` but `aquadoc_core` never imports from here.

```sh
pip install -e ".[research]"
python -m research.lab_matching.cli --help
python -m unittest discover -s research/tests -t .
```

Documentation: `docs/project-audit.md`, `docs/datasets/`, `study.md`.
