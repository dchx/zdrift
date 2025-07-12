End-to-end redshift drift simulation tool

## Spectral Simulation

### Generate Lyman alpha forest spectra

#### From Keck HIRES Data
- `read_koa.py` - read KOA extracted spectra
- `continuum_fit.py` - fit continuum and normalize Keck spectra

#### From parameter distributions
- `generate_spec.py`

### Fit Voigt profiles
- `vamp.py` - find regions to fit
- `voigtforest.py` - fit multiple Voigt profiles

### Estimate sigma

#### Line-wise sigma
- `sse.py`

#### Pixel-wise sigma
- `liske_sigma.py`

## Analysis
- `realll_vs_simll.py` - compare real line lists vs simulated line lists
- `explore_setup.py` - sigma vs z\_QSO, sigma vs resolution
- `target_selection.py` - target selection
- `simulation_setup.py` - simulation setup
- `obs_plan.py` - observation plan over one year
- `vds_aperture_period.py` - vdot/sigma vs aperture and time

## Other tools
- `utils.py` - general tools
- `spec_utils.py` - play with spectra
- `cosmology.py` - cosmology related tools
- `flux_unit_change.py` - change flux units
- `m1450.py` - M\_1450 flux conversions
- `read_elqs.py`, `read_ps.py`, `read_sdss.py` - read ELQS, PS-ELQS, SDSS data

## Citation
[Forecasting cosmic acceleration measurements using the Lyman-α forest](https://doi.org/10.1093/mnras/stac1702)

```
@ARTICLE{2022MNRAS.514.5493D,
       author = {{Dong}, Chenxing and {Gonzalez}, Anthony and {Eikenberry}, Stephen and {Jeram}, Sarik and {Likamonsavad}, Manunya and {Liske}, Jochen and {Stelter}, Deno and {Townsend}, Amanda},
        title = "{Forecasting cosmic acceleration measurements using the Lyman-{\ensuremath{\alpha}} forest}",
      journal = {\mnras},
     keywords = {intergalactic medium, quasars: absorption lines, cosmology: miscellaneous, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = aug,
       volume = {514},
       number = {4},
        pages = {5493-5505},
          doi = {10.1093/mnras/stac1702},
archivePrefix = {arXiv},
       eprint = {2206.08042},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.5493D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
