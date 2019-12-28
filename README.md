# openbci-fmaps

## Dependencies

- cuda-10.2
- [nvidia machine learning repos](http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/)
- anaconda3

## Setup instructions

1. Pull submodules
`submodule update --init --recursive`

2. Create and activate our conda environment
`conda env create --file environment.yml && conda activate openbci-fmaps`

3. Install pycyda
`cd cuda-packages/pycuda/ && python setup.py build && python setup.py install && python test/test_gpuarray.py && cd ../../`

4. Install scikit-cuda
`cd cuda-packages/scikit-cuda/ && python setup.py install && python setup.py test && cd ../../`

## Run
`python run.py`
