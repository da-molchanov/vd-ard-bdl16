from __future__ import print_function

import theano
import lasagne
import numpy as np
from theano import function
from theano import tensor as T
from lasagne.utils import one_hot
from sklearn.metrics import accuracy_score as acc
from lasagne.objectives import categorical_crossentropy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class vdrvc(object):
    def __init__(self):
        self._srng = RandomStreams(42)
        self.theta = None
        self.log_alpha = None

    def score(self, X, t):
        return acc(np.argmax(X.dot(self.theta.T), axis=1), t)

    def predict(self, X):
        return np.argmax(X.dot(self.theta.T), axis=1)

    def fit(self, X, t, num_classes, batch_size, max_iter=1000, display_each=100, lr=1e-2, beta=0.95):
        N, d = X.shape

        def create_theano_loss(d):
            X, t = T.dmatrix('X'), T.dvector('t')
            log_sigma2 = theano.shared(np.ones((num_classes, d)))
            theta = theano.shared(np.random.randn(num_classes, d))

            # Change parametrization
            log_alpha = log_sigma2 - T.log(theta ** 2)
            la, alpha = log_alpha, T.exp(log_alpha)

            # -KL(q || prior)
            mD_KL = -(0.5 * T.log1p(T.exp(-la)) - (0.03 + 1.0 / (1.0 + T.exp(-(1.5 * (la + 1.3)))) * 0.64)).sum()

            # NLL through Local Reparametrization
            mu, si = T.dot(X, theta.T), T.sqrt(T.dot(X*X, (alpha * theta * theta).T))
            activation = mu + self._srng.normal(mu.shape, avg=0, std=1) * si
            predictions = T.nnet.softmax(activation)
            ell = -T.sum(categorical_crossentropy(predictions, one_hot(t, num_classes)))

            # Objective Negative SGVLB
            nlb = - (N/batch_size * ell + mD_KL)

            # Optimization Method and Function Compiling
            opt = lasagne.updates.adam(nlb, [log_sigma2, theta], learning_rate=lr, beta1=beta)
            lbf = function([X, t], nlb, updates=opt)

            return lbf, theta, log_sigma2

        lbf, theta, log_sigma2 = create_theano_loss(d)

        # Main loop
        for i in range(max_iter):
            if batch_size != N:
                idx = np.random.choice(X.shape[0], batch_size)
                loss = lbf(X[idx], t[idx])
            else:
                loss = lbf(X, t)

            if display_each and i % display_each == 0:
                self.theta = theta.get_value()
                self.log_alpha = log_sigma2.get_value() - 2 * np.log(np.abs(self.theta))
                acc_, ard_ = acc(self.predict(X), t), np.sum(self.log_alpha > 5) * 1.0 / self.log_alpha.size
                print('iter = %.4f' % i, 'vlb = %.4f' % loss, 'acc = %.4f' % acc_, 'ard = %.4f' % ard_)

        return self

