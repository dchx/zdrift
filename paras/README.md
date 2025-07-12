To read the parameter files:
```python
import utils
lam0, lgNHI, b = utils.pkloadgzip(para_file)
```
To generate spectra from parameter files:
```python
import utils
import generate_spec as gs
parray = utils.pkloadgzip(para_file)
lam = ... # wavelength points
flux = gs.parray2flux_shift(parray, lam, ...)
```
