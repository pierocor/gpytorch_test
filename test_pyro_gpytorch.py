
"""

Latent Function Inference with Pyro + GPyTorch Test code.

Adapted from:
https://docs.gpytorch.ai/en/v1.5.1/examples/07_Pyro_Integration/Pyro_GPyTorch_Low_Level.html

"""
import sys
import math
import torch
import pyro
import gpytorch
import numpy as np

import argparse
from time import perf_counter


parser = argparse.ArgumentParser()
parser.add_argument('--n_data', dest='n_data', action='store', help='number observations', type=int, default=100)
parser.add_argument('--num_iter', dest='num_iter', action='store', help='number of iterations', type=int, default=1)
parser.add_argument('--num_tests', dest='num_tests', action='store', help='number of tests to perform', type=int, default=1)
parser.add_argument('--num_particles', dest='num_particles', action='store', help='number of particles', type=int, default=1)
parser.add_argument('--num_inducing', dest='num_inducing', action='store', help='number of inducing points', type=int, default=64)
parser.add_argument('--seed', dest='seed', action='store', help='RNG seed', type=int, default=0)
parser.add_argument('--vectorize_particles', dest='vectorize_particles', action='store_true', help='Vectorise particles', default=False)

class PVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing=64, name_prefix="mixture_gp"):
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points = torch.linspace(0, 1, num_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):

        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Use the link function to convert GP samples into scale samples
            scale_samples = function_samples.exp()

            # Sample from observed distribution
            return pyro.sample(
                self.name_prefix + ".y",
                pyro.distributions.Exponential(scale_samples.reciprocal()),  # rate = 1 / scale
                obs=y
            )

def main_func():
    args = parser.parse_args()
    n_data = args.n_data
    num_tests = args.num_tests
    num_iter = args.num_iter
    num_particles = args.num_particles
    num_inducing = args.num_inducing
    vectorize_particles = args.vectorize_particles

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Here we specify a 'true' latent function lambda
    scale = lambda x: np.sin(2 * math.pi * x) + 1

    X = np.linspace(0, 1, n_data)
    Y = np.zeros_like(X)

    for i,x in enumerate(X):
        Y[i] = np.random.exponential(scale(x), 1)

    train_x = torch.tensor(X).float()
    train_y = torch.tensor(Y).float()



    tot_time = []
    for j in range(num_tests):
        interrupted=False

        model = PVGPRegressionModel(num_inducing=num_inducing)

        optimizer = pyro.optim.Adam({"lr": 0.1})
        elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=vectorize_particles, retain_graph=True)
        svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

        model.train()

        t_start = perf_counter()
        for i in range(num_iter):

            model.zero_grad()

            try:
                loss = svi.step(train_x, train_y)
            except Exception as e:
                print(f"ERROR at iteration {i}!", file=sys.stderr)
                print(f"n_data: {n_data} num_iter: {num_iter} num_tests: {num_tests}" +
                      f" num_particles:{num_particles} num_inducing:{num_inducing}" +
                      f" vectorize_particles:{vectorize_particles} seed:{seed}", file=sys.stderr)
                print(e, file=sys.stderr)
                interrupted=True
                break

        t_stop = perf_counter()
        if not interrupted:
            elapsed = t_stop - t_start
            tot_time.append(elapsed)

    print(f"{n_data}\t{len(tot_time)}\t{np.average(tot_time):.2f}\t{np.std(tot_time):.2f}")

if __name__ == "__main__":
    main_func()
