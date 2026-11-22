"""Explicit losses/gradients; no PUF secrets or test data enter training."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .puf import features


def logistic_loss(w, x, y, l2):
    z = x @ w
    penalty = w.copy()
    penalty[-1] = 0
    return (np.mean(np.logaddexp(0, z)-y*z) + l2*np.dot(penalty, penalty)/2,
            x.T @ (expit(z)-y)/len(y)+l2*penalty)


def product_loss(flat, x, y, k, l2):
    """Smooth XOR predictor E[r] = product(tanh(X w_j))."""
    w = flat.reshape(k, x.shape[1])
    h = np.tanh(x @ w.T)
    mu = np.prod(h, axis=1)
    p = (1+mu)/2
    # Add epsilon smoothly, rather than clipping: preserves analytic gradient.
    eps = 1e-9
    loss = -np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
    dmu = .5*((1-y)/(1-p+eps)-y/(p+eps))/len(y)
    grad = np.empty_like(w)
    for j in range(k):
        others = np.prod(np.delete(h, j, axis=1), axis=1)
        grad[j] = x.T @ (dmu*others*(1-h[:, j]**2))
    return loss+l2*np.sum(w*w)/2, (grad+l2*w).ravel()


def mlp_loss(flat, x, y, width, l2):
    d = x.shape[1]
    w1 = flat[:d*width].reshape(d, width)
    b1 = flat[d*width:(d+1)*width]
    w2 = flat[(d+1)*width:-1]
    b2 = flat[-1]
    h = np.tanh(x @ w1+b1)
    z = h @ w2+b2
    dz = (expit(z)-y)/len(y)
    dh = dz[:, None]*w2*(1-h*h)
    loss = np.mean(np.logaddexp(0, z)-y*z)+l2*(np.sum(w1*w1)+np.sum(w2*w2))/2
    grad = np.concatenate(((x.T @ dh+l2*w1).ravel(), dh.sum(axis=0),
                           h.T @ dz+l2*w2, [dz.sum()]))
    return loss, grad


class Attack:
    def __init__(self, kind="lr", k=1, seed=0, epochs=100, restarts=3, width=32):
        if kind not in {"lr", "raw_lr", "xor_lr", "mlp", "majority"}:
            raise ValueError("unknown attack")
        self.kind, self.k, self.seed = kind, k, seed
        self.epochs, self.restarts, self.width = epochs, restarts, width

    def _x(self, c):
        return np.column_stack((c, np.ones(len(c)))) if self.kind == "raw_lr" else features(c)

    def fit(self, c, y, val_c, val_y):
        x, vx = self._x(c), self._x(val_c)
        y, val_y = np.asarray(y, float), np.asarray(val_y, float)
        if len(y) != len(x) or not np.all((y >= 0) & (y <= 1)):
            raise ValueError("targets must be probabilities in [0, 1]")
        self.trace = []
        self.converged = None
        if self.kind == "majority":
            self.constant = float(np.mean(y) >= .5)
            return self
        if self.kind in {"lr", "raw_lr"}:
            fit = minimize(logistic_loss, np.zeros(x.shape[1]), args=(x,y,1e-5),
                           jac=True, method="L-BFGS-B", options={"maxiter":400})
            self.params = fit.x
            self.converged = bool(fit.success)
            return self
        rng = np.random.default_rng(self.seed)
        best = np.inf
        for restart in range(self.restarts):
            if self.kind == "xor_lr":
                params = rng.normal(scale=.5/np.sqrt(x.shape[1]), size=self.k*x.shape[1])
                objective = lambda w, a, b: product_loss(w,a,b,self.k,1e-5)
            else:
                d, h = x.shape[1], self.width
                params = np.concatenate((rng.normal(scale=1/np.sqrt(d),size=d*h),
                                         np.zeros(h),rng.normal(scale=1/np.sqrt(h),size=h),[0.]))
                objective = lambda w, a, b: mlp_loss(w,a,b,self.width,1e-5)
            moment = np.zeros_like(params)
            variance = np.zeros_like(params)
            step, stale, local_best = 0, 0, np.inf
            for epoch in range(self.epochs):
                order = rng.permutation(len(y))
                for start in range(0,len(y),256):
                    idx = order[start:start+256]
                    _, g = objective(params,x[idx],y[idx])
                    step += 1
                    moment = .9*moment+.1*g
                    variance = .999*variance+.001*g*g
                    params -= .01*(moment/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8)
                loss, _ = objective(params,vx,val_y)
                self.trace.append({"restart":restart,"epoch":epoch+1,"validation_loss":float(loss)})
                if loss < best:
                    best = loss
                    self.params = params.copy()
                if loss < local_best-1e-5:
                    local_best, stale = loss, 0
                else:
                    stale += 1
                if stale >= 18:
                    break
        return self

    def probability(self,c):
        if self.kind == "majority":
            return np.full(len(c), self.constant)
        x = self._x(c)
        if self.kind in {"lr", "raw_lr"}:
            return expit(x @ self.params)
        if self.kind == "xor_lr":
            return (1+np.prod(np.tanh(x @ self.params.reshape(self.k,-1).T),axis=1))/2
        d, h = x.shape[1], self.width
        w = self.params
        hidden = np.tanh(x @ w[:d*h].reshape(d,h)+w[d*h:(d+1)*h])
        return expit(hidden @ w[(d+1)*h:-1]+w[-1])

    def predict(self,c):
        return np.where(self.probability(c) >= .5,1,-1).astype(np.int8)

    def save(self,path):
        np.savez_compressed(path,kind=self.kind,k=self.k,seed=self.seed,width=self.width,
                            params=getattr(self,"params",np.array([])),
                            constant=getattr(self,"constant",np.nan))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            model = cls(str(z['kind']),int(z['k']),int(z['seed']),width=int(z['width']))
            model.params = z['params'].copy()
            model.constant = float(z['constant'])
        return model
