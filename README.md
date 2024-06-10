# Quick Installation
Before you begin, ensure that you have Python installed on your system. ```[pinn_clusters](https://github.com/YaoGroup/pinn_clusters)``` [pinn_clusters](https://github.com/YaoGroup/pinn_clusters) is compatible with Python 3.x. To install ```pinn_clusters```, simply run the following command in your terminal:
```
python -m pip install " pinn_clusters @ git+https://github.com/YaoGroup/ice_1D_pinn.git"
```
To install the specific version used in the paper.
```
python -m pip install " pinn_clusters @ git+https://github.com/YaoGroup/ice_1D_pinn.git@379852f184dead0ef7f2f7e97bfbc9dc47e3407e" 
```
# Importing Functions
After installing ```pinn_clusters```, you can import any function from the package directly into your Python script. Just like importing function from other Python libraries like TensorFlow, e.g. ```from tensorflow import keras```, you can importing a function from ```pinn_clusters``` using 
```
from pinn_clusters import your_function_here
```

Replace your_function_here with the specific function you wish to use from the ```pinn_clusters``` package. For instance, to use the gamma_batch function, simply add this line to your code: 

```
from pinn_clusters import gamma_batch
```

# Local Installation

For usage on your local computer, we recommend the use of a conda environment. To install conda, please follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then, create a conda environment with Jupyter installed  by running the command:
```
conda create --name pinn_test python=3.10 jupyter
```
Then, activate the new environment and start jupyter.
```
conda activate pinn_test
jupyter-lab
```
Then, open the example notebook within the jupyter-lab window.

# Code Description
## ```_loss.py```

Contains the loss functions used in the paper, namely ```SquareLoss``` used for fixed collocation points, and ```SquareLossRandom``` used for collocation resampling. Please refer to the final section of this README (__"Code Implementation of Collocation Resampling"__) for a detailed explanation of how these two functions differ.

Both loss functions evaluate the predictive accuracy of the neural network after each iteration according to the characteristic objective function of PINN, which we call $J(\Theta)$: ([Raissi et. al, 2019](https://doi.org/10.1016/j.jcp.2018.10.045))

<p align="center">
$J(\Theta) = \gamma E(\Theta) + (1-\gamma)D(\Theta)$
</p>

where we introduce an additional hyperparameter $\gamma \in [0.0, 1.0]$ to adjust the relative weighting between the equation loss $E(\Theta)$ and the data loss $D(\Theta)$. $D(\Theta)$ is evaluated at values in the domain where training data is available, while $E(\Theta)$ is evaluated at a set of collocation points sampled _independently_ of the available training data. Please see p. 3 of the main text for the precise definitions of equation and data loss used in our paper; p. 5 of the main text for the governing physics equations enforced by $E(\Theta)$, and ```_formulations.py``` for the implementation of these equations in our codes.

### Initialization
An instance of the ```SquareLoss``` function is initialized by the following code:
```
loss = SquareLoss(equations=physics_equations, equations_data=data_equations, gamma=gamma)
```
where
* ```equations```: An iterable of callables with the signature ```function(x, neuralnet)``` corresponding to the governing physics equations. To enforce 1D SSA, we pass ```inverse_1st_order_equations``` imported from ```_formulations.py``` .
*  ```equations_data```: An iterable of callables with the signature ```function(x, neuralnet)``` corresponding to the governing physics equations. We use ```data_equations``` imported from ```_formulations.py```.
*  ```gamma``` (float): the value of $\gamma$ with which to evaluate the objective function $J(\Theta)$.

```SquareLossRandom``` is initialized with the same arguments.
## ```_formulations.py```
Implements ```data_equations``` and ```inverse_1st_order_equations```, which serve as the two components of the cost function. Also implemented are the following helper functions:
  * ```analytic_h_constantB(x)```: analytic $h(x)$ solution for constant $B(x)$ profile.
  * ```analytic_u_constantB(x)```: analytic $u(x)$ solution for constant $B(x)$ profile.
  * ```get_collocation_points(x_train, xmin: float, xmax: float, N_t: int)```: generates a single set of collocation points. 

## ```_constants.py```
Defines the values of the physical constants that appear in the physics-enforcing equations.

## ```_model.py```
Helper functions for neural network initialization.

## ```_optimization.py```
Implements Adam and L-BFGS optimizers.

# Code Implementation of Collocation Resampling

Training can be switched between using fixed collocation points and collocation resampling by switching the loss function used during training. The loss function evaluated by a given optimizer is specified during the initialization of the optimizer. Use the  ```SquareLoss``` loss function when using fixed collocation points, and ```SquareLossRandom``` for random collocation resampling (see lines 77-100 in 'pinn_trials.py').

Comparing the ```SquareLoss``` and ```SquareLossRandom``` functions in 'loss.py', the main difference between the two functions is in the ```__call__``` method. For ```SquareLossRandom```, we add a few extra lines at the beginning of the  ```__call__``` method (lines 54-61):

```
def __call__(self, x_eqn, data_pts, net) -> Dict[str, tf.Tensor]:
    xmin = 0.0
    xmax = 1.0
    N_t = 1001
    _data_type = tf.float64       
    collocation_pts = xmin + (xmax - xmin) * self.col_gen.uniform(shape = [N_t])
    collocation_pts = collocation_pts**3
```
where ```self.col_gen``` is a stateful random generator defined in the ```__init__``` method (line 52):

```
        self.col_gen = tf.random.get_global_generator()
```
Thus, the ```SquareLossRandom``` function generates a new set of collocation points every time it is called, i.e. at every iteration. 

__Important Note: It is essential to use a _stateful_ random number generator such as ```tf.random.Generator()``` to ensure that the collocation points are resampled after each iteration.__ Using a stateless random generator (such as 
 those provided in the ```numpy.random``` module, or the ```lhs``` generator used in our codes for fixed collocation point generation) will not allow the collocation points to be updated in a TensorFlow training loop, causing the loss function to behave identically to training with fixed collocation points.

 # Citation
Yunona Iwasaki and Ching-Yao Lai.
*One-dimensional ice shelf hardness inversion: Clustering behavior and collocation resampling in physics-informed neural networks.* Journal of Computational Physics, Volume 492, 2023, 112435, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2023.112435.

**BibTex:**
```
@article{IWASAKI2023112435,
          title = {One-dimensional ice shelf hardness inversion: Clustering behavior and collocation resampling in physics-            informed neural networks},
          journal = {Journal of Computational Physics},
          volume = {492},
          pages = {112435},
          year = {2023},
          issn = {0021-9991},
          doi = {https://doi.org/10.1016/j.jcp.2023.112435},
          url = {https://www.sciencedirect.com/science/article/pii/S0021999123005302},
          author = {Yunona Iwasaki and Ching-Yao Lai},
          keywords = {Physics-informed neural networks, Scientific machine learning, Ice dynamics, Geophysical fluid                   dynamics, Nonlinear dynamics, Inverse problems},
      }
```
