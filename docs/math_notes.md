# Math Notes: Surprise-Gated TTT + Anchored Forgetting

## Discrete update

Let inner loss be \(\ell_t(\theta)\). Define

- gradient \(g_t = \nabla \ell_t(\theta_t)\)
- surprise \(s_t = \|g_t\|\)
- EMA: \(\tilde s_{t+1} = \alpha \tilde s_t + (1-\alpha) s_t\)
- gate: \(\gamma_t = \eta \cdot \mathrm{clip}(\tau/(\tilde s_t+\varepsilon), 0, 1)\)

Anchored (retention) regularizer:
\[
R_t(\theta)=\tfrac{\lambda_t}{2}\|\theta-\theta_0\|^2
\]

Coupled update:
\[
\theta_{t+1} = \theta_t - \gamma_t(\nabla \ell_t(\theta_t) + \lambda_t(\theta_t-\theta_0))
\]

Equivalently, gradient descent on \(J_t(\theta)=\ell_t(\theta)+R_t(\theta)\) with step \(\gamma_t\).

## Local quadratic stability

If \(\ell\) is locally quadratic around \(\theta^\*\):
\[
\nabla \ell(\theta) \approx H(\theta-\theta^\*)
\]
and \(\lambda_t\equiv \lambda\), \(\gamma_t\equiv \gamma\) near the fixed point, then
\[
e_{t+1}=(I-\gamma(H+\lambda I))e_t
\]
so stability requires \(0<\gamma<2/(\lambda_{\max}(H)+\lambda)\).
Non-oscillatory contraction is guaranteed if \(0<\gamma\le 1/(\lambda_{\max}(H)+\lambda)\).

## EMA time constant

EMA has time constant \(T\approx 1/(1-\alpha)\).
If surprise has correlation time \(T_c\), pick \(\alpha\approx e^{-1/T_c}\).
