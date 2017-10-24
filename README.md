# Deep Learning - The Straight Dope

This is staging repo for https://github.com/zackchase/mxnet-the-straight-dope

The reason for creating another repo is

- Easy to setup CI
- Not introduce too many errors into the original repo
- Will merge back changes once in a good shape

We made the following changes

- Changed to `.ipynb` to `.md` format to simplify code merging.
- Releasing a zip file with all converted `.ipynb` files at the same time
- Building a pdf file through latex
- Hosting the website on S3 with CloudFront
- Testing every notebook in CI, which has 2 M60 GPUs. The timeout for each notebook is 20min


Prebuild doc is at http://tutorials.gluon.ai
