#! /bin/bash

source lazy/bin/activate
python -c "import gpytorch; print(gpytorch.__version__, gpytorch.__file__)"
python -c "import pyro; print(pyro.__version__, pyro.__file__)"

num_tests=1
num_iter=10
n_data=10000
num_particles=32
num_inducing=64
seed=0
vectorize_particles="True"

echo "### num_tests ${num_tests} num_iter ${num_iter} num_particles ${num_particles}"
echo "### num_inducing ${num_inducing} vectorize_particles ${vectorize_particles} seed ${seed}"

if [ $vectorize_particles == "True" ]; then
    vectorize="--vectorize_particles"
else
    vectorize_particles="False"
    vectorize=""
fi

for LAZY in 0 1; do
    if [ $LAZY -eq 1 ]; then
        export GPYTORCH_LAZY_JITTER=""
        echo "### lazy add_jitter"
    else
        unset GPYTORCH_LAZY_JITTER
        echo "### original add_jitter"
    fi

    echo -e "N_data\tN_tests\ttime(avg)\ttime(stddev)\tMemory(KB)"
    for n_data in 100 {10000..100000..10000}; do
        /usr/bin/time -f "%M" \
            python -W ignore test_pyro_gpytorch.py \
                --n_data ${n_data} \
                --num_tests ${num_tests} \
                --num_iter ${num_iter} \
                --num_particles ${num_particles} \
                --num_inducing ${num_inducing} \
                --seed ${seed} \
                ${vectorize} \
        2>&1 | sed -rz 's/(.*)\n([^\n]*\n)/\1\t\2/'
    done
done
