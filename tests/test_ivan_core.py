import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import check_grad
from puf_attacks.puf import ArbiterPUF, features, split_challenges
from puf_attacks.models import Attack, logistic_loss, product_loss, mlp_loss


class SimulatorTests(unittest.TestCase):
    def test_feature_convention(self):
        np.testing.assert_array_equal(features(np.array([[1,-1,1],[-1,-1,-1]])),
                                      [[-1,-1,1,1],[-1,1,-1,1]])

    def test_independent_stage_recurrence(self):
        # delta_i = c_i delta_(i-1) + a_i c_i + b_i;
        # expanding gives w_i = a_i+b_(i-1), w_0=a_0, bias=b_last.
        rng = np.random.default_rng(81)
        c = rng.choice([-1,1],(200,16))
        a,b = rng.normal(size=(2,16))
        w = np.r_[a[0],a[1:]+b[:-1],b[-1]]
        delta = np.zeros(len(c))
        for i in range(16):
            delta = c[:,i]*delta+a[i]*c[:,i]+b[i]
        np.testing.assert_allclose(features(c)@w,delta,atol=1e-12)

    def test_xor_composition_and_seeds(self):
        p = ArbiterPUF.random(16,3,5)
        c = np.random.default_rng(6).choice([-1,1],(100,16))
        expected = np.prod([ArbiterPUF(w[None,:]).response(c)[:,0] for w in p.weights],axis=0)
        np.testing.assert_array_equal(p.response(c)[:,0],expected)
        np.testing.assert_array_equal(p.weights,ArbiterPUF.random(16,3,5).weights)
        np.testing.assert_array_equal(p.response(c,.4,3,np.random.default_rng(7)),
                                      p.response(c,.4,3,np.random.default_rng(7)))

    def test_exact_noise_probability(self):
        p = ArbiterPUF(np.array([[.5,.1],[-.3,.4]]))
        c = np.array([[1],[-1]])
        mean = np.mean(p.response(c,.5,100000,np.random.default_rng(8)) == 1,axis=1)
        np.testing.assert_allclose(mean,p.probability_positive(c,.5),atol=.006)

    def test_splits_disjoint(self):
        splits = split_challenges(12,[100,50,50],1)
        keys = [{r.tobytes() for r in s} for s in splits]
        self.assertEqual(len(set.union(*keys)),200)
        with self.assertRaises(ValueError):
            features([[0,1]])


class ModelTests(unittest.TestCase):
    def test_all_gradients(self):
        rng=np.random.default_rng(9)
        x=np.c_[rng.normal(size=(12,4)),np.ones(12)]
        y=rng.uniform(size=12)
        cases=[(lambda w:logistic_loss(w,x,y,.02),5),
               (lambda w:product_loss(w,x,y,3,.02),15),
               (lambda w:mlp_loss(w,x,y,3,.02),22)]
        for objective,size in cases:
            w=rng.normal(scale=.2,size=size)
            error=check_grad(lambda z:objective(z)[0],lambda z:objective(z)[1],w)
            self.assertLess(error,2e-6)

    def test_soft_labels_equal_repeated_bce(self):
        x=np.array([[1.,1.],[-1.,1.]])
        labels=np.array([[0.,1.,1.],[0.,0.,1.]])
        w=np.array([.4,-.2])
        a=logistic_loss(w,x,labels.mean(axis=1),.01)
        b=logistic_loss(w,np.repeat(x,3,axis=0),labels.ravel(),.01)
        self.assertAlmostEqual(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1])

    def test_learns_and_roundtrips(self):
        p=ArbiterPUF.random(16,1,1)
        train,val,test=split_challenges(16,[1500,300,1000],2)
        model=Attack('lr').fit(train,(p.response(train)[:,0]+1)/2,
                                val,(p.response(val)[:,0]+1)/2)
        self.assertGreater(np.mean(model.predict(test)==p.response(test)[:,0]),.97)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.npz'
            model.save(path)
            np.testing.assert_array_equal(model.predict(test),Attack.load(path).predict(test))

    def test_product_learns_xor(self):
        p=ArbiterPUF.random(16,2,3)
        train,val,test=split_challenges(16,[3000,400,1000],4)
        model=Attack('xor_lr',2,seed=5,epochs=100,restarts=2).fit(
            train,(p.response(train)[:,0]+1)/2,val,(p.response(val)[:,0]+1)/2)
        self.assertGreater(np.mean(model.predict(test)==p.response(test)[:,0]),.9)


class ExperimentTests(unittest.TestCase):
    def test_budget_resume_and_provenance(self):
        from puf_attacks.experiments import run
        config=json.loads(Path('configs/ivan/smoke.json').read_text())
        config.update(chains=[1],models=['lr'],budgets=[301],
                      strategies=[dict(name='majority_3',repeats=3,target='majority')])
        with tempfile.TemporaryDirectory() as d:
            config_path=Path(d)/'config.json'
            config_path.write_text(json.dumps(config))
            out=Path(d)/'results'
            a=run(config_path,out)
            b=run(config_path,out)
            self.assertEqual(len(a),len(b))
            self.assertEqual(a[0]['queries_used'],301)
            config['noise']=[.1]
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(config_path,out)


if __name__ == '__main__':
    unittest.main()
