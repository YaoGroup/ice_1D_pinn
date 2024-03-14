from abc import ABC, abstractmethod
import time
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def _get_indices(trainable_vars):
    # we'll use tf.dynamic_stitch and tf.dynamic_partition,
    # so we need to prepare required information
    shapes = tf.shape_n(trainable_vars)
    count = 0
    stitch_indices = []
    partition_indices = []
    for i, shape in enumerate(shapes):
        n = np.product(shape.numpy())
        stitch_indices.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        partition_indices.extend([i] * n)
        count += n
    partition_indices = tf.constant(partition_indices)
    return partition_indices, stitch_indices


def assign_new_model_parameters(params, training_vars, partition_indices):
    shapes = tf.shape_n(training_vars)
    params = tf.dynamic_partition(params, partition_indices, len(shapes))
    for i, (shape, param) in enumerate(zip(shapes, params)):
        training_vars[i].assign(tf.reshape(param, shape))

class OptimizerBase(ABC):
    """
    Base class for optimizers used in training physics-informed neural networks.

    Attributes:
        net (tf.keras.Model): The neural network model.
        loss (Callable): Function to compute the loss.
        colloc_pts (tf.Tensor): Collocation points for physics-informed part.
        data_pts (Tuple[tf.Tensor, tf.Tensor]): Tuple of x and y data points.
        _loss_records (Dict): Dictionary to store loss records.
        history (List): List to store history of losses.
    """

    def __init__(self, net, loss, collocation_points, data_points) -> None:
        self.net = net
        self.loss = loss
        self.colloc_pts = collocation_points
        self.data_pts = data_points if data_points else (None, None)
        self._loss_records = {}
        self.history = []

    @abstractmethod
    def loss_records(self) -> Dict[str, np.array]:
        """
        Abstract method to return loss records.
        """
        pass

    @abstractmethod
    def optimize(self, nIter: int) -> None:
        """
        Abstract method to perform optimization.
        
        Args:
            nIter (int): Number of iterations for the optimization.
        """
        pass

    def _update_loss_records(self, losses: Dict[str, tf.Tensor], iteration: int) -> None:
        """
        Helper method to update loss records.

        Args:
            losses (Dict[str, tf.Tensor]): Dictionary of current losses.
            iteration (int): Current iteration number.
        """
        msg = f"Iter {iteration:4d}; "
        for name, loss in losses.items():
            record = self._loss_records.setdefault(name, [])
            loss_val = loss.numpy()
            record.append(loss_val)
            msg += f"{name}: {loss_val:4e}; "
        print(msg)


class LBFGS(OptimizerBase):

    @property
    def loss_records(self) -> Dict[str, np.array]:
        return self._loss_records

    @tf.function
    def _single_iteration(self, params, partition_indices, stitch_indices):
        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        #add line to generate new set of collocation points; replace self.collo
        ##new code below
        ##col_pts = get_collocation_points(x_train=self.x_train, xmin=self.xmin, xmax=self.xmax, N_t=self.N_t)
        with tf.GradientTape() as tape:
            # update the parameters in the model
            # this step is critical for self-defined function for L-BFGS
            assign_new_model_parameters(params, self.net.trainable_weights, partition_indices)
            losses = self.loss(x_eqn=self.colloc_pts, data_pts=self.data_pts, net=self.net)
            ##losses = self.loss(x_eqn=col_pts, data_pts=self.data_pts, net=self.net)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(losses["loss"], self.net.trainable_weights)
        grads = tf.dynamic_stitch(stitch_indices, grads)

        # store loss value so we can retrieve later
        tf.py_function(self.history.append, inp=[losses["loss"]], Tout=[])
        return losses, grads

    ##add new arguments specifying domain for collocation points.
    def optimize(self, nIter: int):
        self._loss_records = {}
        self.start_time = time.time()

        # move this outside the decorator to save the losses
        iteration = tf.Variable(0)
        partition_indices, stitch_indices = _get_indices(self.net.trainable_weights)

        def optimize_func(params):
            # A function that can be used by tfp.optimizer.lbfgs_minimize.
            # This function is created by function_factory.
            # Sub-function under function of class not need to input self
            losses, grads = self._single_iteration(params, partition_indices, stitch_indices)
            iteration.assign_add(1)

            if iteration % 10 == 0:
                msg = f"LBFGS Iter {iteration.numpy():4d}; "
                for name, loss in losses.items():
                    record = self._loss_records.setdefault(name, [])
                    loss_val = loss.numpy()
                    record.append(loss_val)
                    msg += f"{name}: {loss_val:4e}; "
                print(msg)

            return losses["loss"], grads

        max_nIter = tf.cast(nIter/3, dtype=tf.int32)
        init_params = tf.dynamic_stitch(stitch_indices, self.net.trainable_weights)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=optimize_func,
            initial_position=init_params,
            tolerance=10e-30, max_iterations=max_nIter)

        # after training, the final optimized parameters are still in results.position
        # so we have to manually put them back to the model
        assign_new_model_parameters(results.position, self.net.trainable_weights, partition_indices)


class Adam(OptimizerBase):

    def __init__(self, net, loss, collocation_points, data_points) -> None:
    ##def __init__(self, net, loss, data_points, x_train, xmin, xmax, N_t) -> None:
        super().__init__(net, loss, collocation_points, data_points)
        ##super().__init__(net, loss, data_points, x_train, xmin, xmax, N_t)
        self._adam = tf.optimizers.Adam()

    @property
    def loss_records(self) -> Dict[str, np.array]:
        return self._loss_records

    @tf.function
    def _single_iteration(self):
        ##new code below
        ##col_pts = get_collocation_points(x_train=self.x_train, xmin=self.xmin, xmax=self.xmax, N_t=self.N_t)
        with tf.GradientTape() as tape:
            losses = self.loss(x_eqn=self.colloc_pts, data_pts=self.data_pts, net=self.net)
            ##losses = self.loss(x_eqn=col_pts, data_pts=self.data_pts, net=self.net)
        grads = tape.gradient(losses["loss"], self.net.trainable_weights)
        self._adam.apply_gradients(zip(grads, self.net.trainable_weights))
        return losses

    def optimize(self, nIter: int):
        for it in range(nIter):
            losses = self._single_iteration()

            # Print and store
            if it % 10 == 0:
                msg = f"Adam Iter {it:4d}; "
                for name, loss in losses.items():
                    record = self._loss_records.setdefault(name, [])
                    loss_val = loss.numpy()
                    record.append(loss_val)
                    msg += f"{name}: {loss_val:4e}; "
                print(msg)

#currently reverted changes.
#Changes: --> identifiable by '##'
#1) Completely get rid of object variable self.colloc_pts, wherever it appears
#2) Call get_collocation_pts method in the _single_iteration method for both Adam and LBFGS.
#3) Change the arguments to include the parameters required for get_collocation to work.
#4)get_collocation_points(x_train, xmin: float, xmax: float, N_t: int)
