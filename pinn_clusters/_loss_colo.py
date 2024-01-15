from typing import Dict
import tensorflow as tf


class SquareLossRandom:
    """Calculate square loss from given physics-informed equations and data equations

    Note that Data equation can be used to set boundary conditions too.
    """

    def __init__(self, equations, equations_data, gamma: float) -> None:
        """
        Args:
            equations:
                an iterable of callables, with signatrue function(x, neural_net)
            equations_data:
                an iterable of callables, with signatrue function(x, y, neural_net)
            gamma:
                the normalized weighting factor for equation loss and data loss,
                loss = gamma * equation-loss + (1 - gamma) * data-loss
        """
        self.eqns = equations
        self.eqns_data = equations_data
        self.gamma = gamma
        self.col_gen = tf.random.get_global_generator()

    def __call__(self, x_eqn, data_pts, net) -> Dict[str, tf.Tensor]:
        xmin = 0.0
        xmax = 1.0
        N_t = 1001
        _data_type = tf.float64
        
        collocation_pts = xmin + (xmax - xmin) * self.col_gen.uniform(shape = [N_t])
        collocation_pts = collocation_pts**3 
        
        x_eqn = tf.cast(collocation_pts, dtype=_data_type)
        print(collocation_pts)
        equations = self.eqns(x=x_eqn, neural_net=net)
        x_data, y_data = data_pts
        datas = self.eqns_data(x=x_data, y=y_data, neural_net=net)
        loss_e = sum(tf.reduce_mean(tf.square(eqn)) for eqn in equations)
        loss_d = sum(tf.reduce_mean(tf.square(data)) for data in datas)
        loss = (1 - self.gamma) * loss_d + self.gamma * loss_e
        return {"loss": loss, "loss_equation": loss_e, "loss_data": loss_d, "coldebug" : collocation_pts[-1]}
