# EROS

This algorithm was implemented with regards to the paper written by Stephen Smith [1].


## Usage

```python
from eros import *
import SimpleITK as sitk

image = sitk.ReadImage("Image/Dir.nii.gz")
image = sitk.MaximumProjection(image)
npim = sitk.GetArrayFromImage(image)

out, cx, cy = eros(npim, 2) # The format of the output is [[angle, (com_x, com_y)], ... ]
```

## Prequisite

Requires cv2

## Reference
1. Smith, Stephen, and Mark Jenkinson. "Accurate robust symmetry estimation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Berlin, Heidelberg, 1999.