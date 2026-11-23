import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from puf_lab.attack import objective, fit_product, predict_product
from puf_lab.soft_labels import soft_objective, fit_soft


class SoftLabelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(35)
        self.x = np.column_stack((rng.normal(size=(12, 4)), np.ones(12)))
        self.w = rng.normal(size=(2, 5)).ravel()
        self.bits = rng.integers(0, 2, size=(3, 12))

    def test_fraction_loss_and_gradient_equal_all_measured_response_bits(self):
        q = self.bits.mean(axis=0)
        loss, grad = soft_objective(self.w, self.x, q, 2)
        repeated_x = np.tile(self.x, (3, 1))
        full_loss, full_grad = objective(self.w, repeated_x, 1-2*self.bits.ravel(), 2)
        np.testing.assert_allclose(loss, full_loss, atol=1e-12)
        np.testing.assert_allclose(grad, full_grad, atol=1e-12)

    def test_gradient_including_fractional_targets(self):
        q = self.bits.mean(axis=0)
        _, grad = soft_objective(self.w, self.x, q, 2)
        numeric = approx_derivative(lambda w: soft_objective(w,self.x,q,2)[0], self.w).ravel()
        np.testing.assert_allclose(grad,numeric,atol=1e-8,rtol=1e-6)

    def test_binary_endpoints_agree_with_existing_objective(self):
        for k in (1,2,4):
            w=np.linspace(-.2,.2,k*self.x.shape[1])
            a=soft_objective(w,self.x,self.bits[0],k)
            b=objective(w,self.x,1-2*self.bits[0],k)
            np.testing.assert_allclose(a[0],b[0],atol=1e-12)
            np.testing.assert_allclose(a[1],b[1],atol=1e-12)

    def test_fit_selects_by_validation_and_single_reads_match_hard_fit(self):
        rng=np.random.default_rng(2)
        x=np.column_stack((rng.normal(size=(512,4)),np.ones(512)))
        y=x[:,0]+.5*x[:,1]<0
        a, attempts=fit_soft(x[:256],y[:256],x[256:384],y[256:384],1,np.random.default_rng(6),2,100)
        b,_=fit_product(x[:256],y[:256],x[256:384],y[256:384],1,np.random.default_rng(6),2,100)
        np.testing.assert_array_equal(predict_product(x[384:],a),predict_product(x[384:],b))
        self.assertGreater(np.mean(predict_product(x[384:],a)==y[384:]),.95)
        selected=[t for t in attempts if t['selected']]
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['validation_accuracy'],max(t['validation_accuracy'] for t in attempts))
        with self.assertRaises(ValueError):
            fit_soft(x[:1],[1.2],x[1:2],y[1:2],1,rng,1,1)


if __name__=='__main__':
    unittest.main()
