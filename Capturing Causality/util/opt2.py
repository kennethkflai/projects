"""From built-in optimizer classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip

from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
import numpy as np
from keras.optimizers import Optimizer

class SGDW2(Optimizer):
    """Stochastic gradient descent optimizer with decoupled weight decay.

    Includes support for momentum, learning rate decay, Nesterov momentum,
    and warm restarts.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.

    # References
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.01, momentum=0., decay=0.,
                 nesterov=False, weight_decay=0.025, anneal_lr=True, bs=512.,
                 eta=1.0, steps_per_cycle=1, **kwargs):
        super(SGDW2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle, name='steps_per_cycle')
            self.t_cur = K.variable(0, name='t_cur')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.anneal_lr = anneal_lr
        self.bs = K.cast(bs, 'float32')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        self.updates = [K.update_add(self.t_cur, 1)]

        PI = 3.141592653589793
        fraction_to_restart = K.cast(self.t_cur / (self.steps_per_cycle), 'float32')

        self.eta = 0.5 * (1.0 + K.cos(fraction_to_restart * PI))
        w_d = self.eta*self.weight_decay*K.sqrt(self.bs/self.steps_per_cycle)

        learning_rate = self.learning_rate


        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        if self.anneal_lr:
            learning_rate = self.eta*learning_rate
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - learning_rate * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - learning_rate * g - w_d * p
            else:
                new_p = p + v - w_d * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'eta': float(K.get_value(self.eta)),
                  'steps_per_cycle': int(K.get_value(self.steps_per_cycle))}
        base_config = super(SGDW2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdamW2(Optimizer):
    """AdamW optimizer with decoupled weight decay.
    Default parameters follow those provided in the original Adam paper.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Normalized weight decay.
        eta: float >= 0. The multiplier to schedule learning rate and weight decay.
        steps_per_cycle: int > 0. The number of training batches of a restart cycle.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., weight_decay=0.025, amsgrad=False, bs=512., anneal_lr=True,
                 eta=1.0, steps_per_cycle=1, **kwargs):
        super(AdamW2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.eta = K.variable(eta, name='eta')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.steps_per_cycle = K.variable(steps_per_cycle, name='steps_per_cycle')
            self.t_cur = K.variable(0, name='t_cur')
            self.bs = K.variable( K.cast(bs, 'float32'), name='bs')
            
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.anneal_lr = anneal_lr

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates = [K.update_add(self.t_cur, 1)]

        PI = 3.141592653589793
        fraction_to_restart = K.cast(self.t_cur / (self.steps_per_cycle), 'float32')

        self.eta = 0.5 * (1.0 + K.cos(fraction_to_restart * PI))
        w_d = self.eta*self.weight_decay*K.sqrt(self.bs/self.steps_per_cycle)

        learning_rate = self.learning_rate
        if self.initial_decay > 0:
            learning_rate = learning_rate * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        learning_rate_t = learning_rate * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        if self.anneal_lr:
            learning_rate_t = self.eta*learning_rate_t

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                     dtype=K.dtype(p),
                     name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - learning_rate_t * m_t / (K.sqrt(vhat_t) + self.epsilon) - w_d * p
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - learning_rate_t * m_t / (K.sqrt(v_t) + self.epsilon) - w_d * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.weight_decay)),
                  'eta': float(K.get_value(self.eta)),
                  'steps_per_cycle': int(K.get_value(self.steps_per_cycle)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def _calc_eta(opt):
    PI = 3.141592653589793
    fraction_to_restart = K.cast(opt.t_cur / (opt.steps_per_cycle), 'float32')
    eta = 0.5 * (1.0 + K.cos(fraction_to_restart * PI))
    return eta

class Nadam2(Optimizer):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](
           http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, anneal_lr=True, bs=512.,
                 eta=1.0, steps_per_cycle=1,weight_decay=0.025, **kwargs):
        self.schedule_decay = kwargs.pop('schedule_decay', 0.004)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        self.anneal_lr = anneal_lr
        self.bs = K.cast(bs, 'float32')
        learning_rate = kwargs.pop('lr', learning_rate)
        super(Nadam2, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.t_cur = K.variable(0, name='t_cur')
            self.eta = K.variable(eta, name='eta')
            self.steps_per_cycle = K.variable(steps_per_cycle, name='steps_per_cycle')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates = [K.update_add(self.t_cur, 1)]
        t = K.cast(self.iterations, K.floatx()) + 1

        self.eta = _calc_eta(self)

        w_d = self.eta*self.weight_decay*K.sqrt(self.bs/self.steps_per_cycle)

        learning_rate_t = self.learning_rate
        if self.anneal_lr:
            learning_rate_t = self.eta*self.learning_rate


        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape, name='m_' + str(i))
              for (i, shape) in enumerate(shapes)]
        vs = [K.zeros(shape, name='v_' + str(i))
              for (i, shape) in enumerate(shapes)]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = (p - learning_rate_t * m_t_bar / (K.sqrt(v_t_prime) +
                   self.epsilon))
            new_p = p_t - w_d * p

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include m_schedule at head of the weight list. Set
        # m_schedule to 1.
        if len(params) == len(weights) + 1:
            weights = [weights[0]] + [np.array(1.)] + weights[1:]
        super(Nadam2, self).set_weights(weights)

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Nadam2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))