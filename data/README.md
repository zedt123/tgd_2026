## Datasets

This repository uses small subsets of standard image datasets for non-commercial research and evaluation purposes.

We use the same evaluation images as the DAPS paper. For the original dataset setup and download instructions, please refer to the DAPS repository:

```text
https://github.com/zhangbingliang2019/DAPS/tree/main
```

The expected local directory structure is:

```text
data/
  test-ffhq/
  test-imagenet/
```

## FFHQ

For FFHQ, we use the 100-image evaluation subset provided by the authors of DAPS:

```text
https://drive.google.com/drive/folders/1RHNif32W0hvB4M75ppG1ypTChy-W3q3Z
```

Place the FFHQ images under:

```text
data/test-ffhq/
```

The original FFHQ dataset was curated by NVIDIA Corporation and is licensed under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International license (CC BY-NC-SA 4.0). Users must comply with the FFHQ license terms and provide appropriate attribution.

## ImageNet

ImageNet images are **not redistributed** in this repository due to licensing restrictions.

In our experiments, we used the same 100-image ImageNet evaluation subset as DAPS. Users who wish to reproduce the ImageNet results must obtain ImageNet themselves and agree to the ImageNet terms and conditions:

```text
https://www.image-net.org/
```

Place the ImageNet images under:

```text
data/test-imagenet/
```

## License

The **code** in this repository is licensed under the MIT License.

The **datasets** included or referenced in this repository are subject to their own licenses and terms, which may restrict redistribution and commercial use. Please refer to the dataset-specific sections above for details.