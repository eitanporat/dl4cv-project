# Tasks

- Differentiable FID ❌
2. Differentiable Naive Sampler = Momentum (unrolling) ❌
$$z_t = \begin{pmatrix}x_t \\ f_{\theta}(x_t, t) \\ \epsilon\end{pmatrix}$$
    a. Fully-connected ❌
    b. Convolutional ✅
    c. With/without noise ❌
    d. Skip connection ✅
    e. Learn time-steps ❌
3. Differentiable IS ❌
4. Writing the training loop of the unrolling
    a. Modify the number of time-steps ❌
5. RNN ❌
6. Conditional sampling $\nabla_{x_t} \log \Pr(x_t | y) = \nabla_{x_t} \log \Pr(y="cat" | x_t) + \nabla_{x_t} \log \Pr(x_t) $ ❌
    a. Add gradient of classifier ❌
    b. Add CLIPLoss to loss ❌
    c. batches of specific class ❌

$$z'_t = \begin{pmatrix}x_t \\ f_{\theta}(x_t, t) \\ \epsilon \\ \log g_{\phi}(y="cat" | x_t)\end{pmatrix}$$ ❌
$g_\phi$ is the inception network.

7. Impainting ❌
8. Why is the time embedding important?
9. Create Hybrid Sampler-Gradient Architecture - Diffusion GANs?
10. GAN Loss - use a pretrained cifar10 classifier
11. Train simpler model for diffusion
12. Loss = ||E(x+noise_t, t)-E(x)||_2 + ||D(E(x))-x||_2 where E is an encoder and D is a decoder
13. Model fine-tuning