# GPyTorch test
Test for "Latent Function Inference with Pyro + GPyTorch".

### Install
```
# Clone the repository and update submodules
git clone --recurse-submodules https://github.com/pierocor/gpytorch_test.git
cd gpytorch_test

# Create virtual environment
python -m venv lazy
source lazy/bin/activate

# Install required packages
pip install -r requirements.txt

# Install GPyTorch (editable)
pip install -e gpytorch/
```

### Run
To run a single test:
```
python test_pyro_gpytorch.py
```
To run the suite of tests reporting the memory consumption:
```
./run_test.sh
```
> The latter requires GNU time

### Python script
Code from [GPyTorch documentation](https://docs.gpytorch.ai/en/v1.5.1/examples/07_Pyro_Integration/Pyro_GPyTorch_Low_Level.html).
