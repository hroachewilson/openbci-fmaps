# openbci-fmaps

## Dependencies

- cuda-10.2
- [http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/](nvidia machine learning repos)
- anaconda3
- OpenBCI

## Setup instructions

`submodule update --init --recursive`

`conda env create --file environment.yml && conda activate openbci-fmaps`


`cd cuda-packages/pycuda/ && python setup.py build && python setup.py install && python test/test_gpuarray.py && cd ../../`


`cd cuda-packages/scikit-cuda/ && python setup.py install && python setup.py test && cd ../../`
