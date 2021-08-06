# Redshift drift

## Spectral Simulation

### Generate spectra

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

## Other functions
- `utils.py`
- `cosmology.py`
