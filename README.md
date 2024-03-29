# Deformed2Self: Self-Supervised Denoising for Dynamic Medical Imaging

## Resources

- [MICCAI 2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_3)
- [paper (arxiv)](https://arxiv.org/pdf/2106.12175.pdf)

## Requirements

- python 3.8
- pytorch 1.4
- pyyaml
- tqdm
- scikit-image

## Run Demo

```
python demo.py
```

## Examples

|noisy|denoised|reference|
|---|---|---|
|![](outputs/noisy2.png)|![](outputs/denoised2.png) |![](data/2.png)|
|![](outputs/noisy5.png)|![](outputs/denoised5.png) |![](data/5.png)|
|![](outputs/noisy8.png)|![](outputs/denoised8.png) |![](data/8.png)|

## Citation

```
@inproceedings{xu2021deformed2self,
  title={Deformed2self: Self-supervised denoising for dynamic medical imaging},
  author={Xu, Junshen and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={25--35},
  year={2021},
  organization={Springer}
}
```

## Contact

For questions, please send an email to junshen@mit.edu
