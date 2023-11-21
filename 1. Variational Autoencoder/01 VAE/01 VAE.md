# Variational Autoencoder

![img](.\asset\structure.jpg)

## Introduction

Although the impressive capabilities of MidJourney and DaLLE3 might make it seem effortless, generating images that make sense to humans is an inherently challenging task, let alone producing images that align with text. Therefore, let's begin from the very basics and concentrate on generating 'images' rather than producing random noise devoid of logic and meaning. To put it more formally, let's prioritize **unconditional image generation** initially.

An image can be perceived as a vector in a high-dimensional space. A typical image consists of three color channels: red, green, and blue. Each channel, for example, comprises 256 * 256 pixels. Therefore, we can represent an image as a 3 * 256 * 256-dimensional vector.

![img](.\asset\vector.jpg)

Now, why do some vectors in this space yield 'meaningful' pictures, while others result in noise? The reason lies in the fact that meaningful images must exhibit certain abstract features, such as similar pixel values in nearby regions and smooth lines within the picture. These meaningful images only occupy a small portion of the vector space. Consequently, if we can determine the distribution of meaningful images, denoted as $ P(x)$, we can easily generate new images by sampling from this distribution!

![img](.\asset\Px.jpg)

For deep learning, if we already have a set of images denoting the real images $ X_{source} = \{x_i| i=0,1,2,...\}$, we should develop a model that gives $ P_{learned}(x)$ to approach $ P_{source}(x)$.

## Latent model

Gaining a comprehensive understanding of high-dimensional spaces can be a formidable challenge. Fortunately, not all pixels within an image carry equally meaningful information, and adjacent pixels often contain redundant data. Moreover, the variations in most pixels have a negligible impact on the overall abstract interpretation of an entire picture. In this context, we anticipate the existence of a lower-dimensional vector space, denoted as $Z$, with, for instance, just 10 dimensions. Each vector within this space encapsulates higher-level and more abstract information. Consequently, a single vector may correspond to a category of real-world images within the set $X$. For instance, a vector like [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0] could be indicative of various dog images.

Mapping $ Z$ to $ X$ is no difficulty for deep learning models. Then, we can change the formula:

$$
P(x) = \int P(x|z;\theta)P(z)dz
$$

Here, $θ$ represents the parameters of a neural network, denoted as $f$, which maps a vector $z$ from $Z$ to $X$. For each $x \in X_{source}$, the objective is to maximize the likelihood $P(x)$. Calculating $P(x|z;\theta)$ is facilitated through the reconstruction loss of the function $f$.

In practical applications, it's common to replace $P(x|z;\theta)$ with a Gaussian distribution $N(x|f(z;\theta), \sigma^2 * I)$ to introduce diversity into the results.

## Autoencoder

Then, why should we change our focus to the latent space $ Z$? We know its dimension is smaller, and contains more useful information. That's good. But what's the goodness for calculation? What id $ P(z)$ and how do we calculate the integral?

Here's the point: We can't change image space $ X$ because it's defined by the dataset (or by nature if you like), but we can assign what distribution is our latent space $ Z$. This is because a complex enough $ f$ will map it to any specific distribution theoretically. For simplicity, we define $ Z = N(0, I)$.

Now that we've resolved the issue of sampling $z$, let's address the challenge of integration. Calculating an integral can be computationally expensive. On the other hand, when given an image $x_i$, only a subset of $z$s will likely result in a high probability reconstruction of it. If we can identify these relevant $z$s, we can somewhat replace the integral with $P(x) = \sum_{i} P(x|z_i;\theta)$.

![img](.\asset\autoencoder.jpg)

This is precisely why we turn to the concept of autoencoders. In an autoencoder, we use an Encoder to map $X_{source}$ to $Z$ and then employ a Decoder to reconstruct $X_{source}$ from $Z$, thereby aligning $X_{source}$ and $Z$.

In VAE, we still need a network $ g(x;\theta^{*})$ mapping $ X$ to $ Z$, besides the decode network $ f(z;\theta)$. In the language of Probability Theory, we need to know the distribution $ Q(z|x)$, which tells us which latents may generate images similar to the given $ x$.

## Variational Autoencoder

One of the most significant departures from traditional autoencoders (AE) in Variational Autoencoders (VAE) is the pre-assigned nature of the latent space, as opposed to learning it during training. To bridge this gap, we need to delve into two fundamental mathematical concepts: Bayes' theorem and Kullback-Leibler (KL) divergence. If you are unfamiliar with these concepts, you can click the link below for a more detailed explanation. However, understanding these principles should not pose a significant barrier to grasping the essence of this deep learning model.

### Bayes Theorem

$$
P(A|B) = \frac{P(B|A)P(A)} {P(B)}
$$

Here's what each term in the formula represents:

1. P(A|B): This is the posterior probability, the probability of event A occurring given that event B has occurred. It represents your updated belief in A after considering the new evidence B.
2. P(B|A): This is the conditional probability of B given A, which represents the probability of observing evidence B when event A is true.
3. P(A): This is the prior probability of A, which represents your initial belief or the probability of A being true before considering any new evidence.
4. P(B): This is the prior probability of B, which represents your initial belief or the probability of B being true before considering any new evidence.

Transform the equation into $ P(A|B)P(B) = P(B|A)P(A)$ may be easier to understand.

*More resources:* [Bayes&#39; Theorem (mathsisfun.com)](https://www.mathsisfun.com/data/bayes-theorem.html)

### Kullback–Leibler divergence (KL divergence)

The Kullback-Leibler divergence, also known as relative entropy, is a measure of how one probability distribution differs from another. It's a concept from information theory and statistics and is often used in various fields, including machine learning, data science, and information retrieval.

The KL divergence between two discrete probability distributions $ P$ and $ Q$ defined on the same space, often denoted as $ D_{KL}(P||Q)$, can be  written as follows:

$$
D_{KL}(P||Q) = \sum \limits_{x \in X}P(x)log\frac{P(x)}{Q(x)}
$$

because $ P(x)$ denotes the probability of $ x$, the equation can be rewritten in the language of expectation:

$$
D_{KL}(P||Q) = E_{x \sim  X} [log(P(x))-log(Q(x))]
$$

*More resources:* [Kullback-Leibler Divergence Explained — Count Bayesie](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

By employing the mathematical tools mentioned above, we will derive the equation. I will provide an explanation after presenting all the steps.

$$
D_{KL}(Q(z|x)||P(z|x)) = E_{z \sim Q}[logQ(z|x) - logP(z|x)] \tag{1}
$$

$$
D_{KL}(Q(z|x)||P(z|x)) = E_{z \sim Q}[logQ(z|x) - log\frac{P(x|z)P(z)}{P(x)}]
$$

$$
D_{KL}(Q(z|x)||P(z|x)) = E_{z \sim Q}[logQ(z|x) - logP(x|z) - logP(z)] + P(x) \tag{2}
$$

$$
P(x) - D_{KL}(Q(z|x)||P(z|x)) = E_{z \sim Q}[logP(x|z)] - D_{KL}(Q(z|x)||P(z)) \tag{3}
$$

We derive the equation under the prior knowledge that $D_{KL}(Q(z|x)||P(z|x))$ should be small because  they are both nice estimation of the map from image space$ X_{source}$ to latent space $ Z$. The only difference is that $ Q(z|x)$ is learned by neural network, while $ P(z|x)$ is the conditional probability derived from Bayes Rules. Thus we first write out (1) as our analyzing object.

Note that $ P(z|x)$ is hard to calculate while $ P(x|z)$ can be estimated through the reconstruction loss of the decoder network. Therefore we apply Bayes Rules to equation (1) and get (2). $ P(x)$ gets out because it's irrelevant to $ z$.

Then $ logQ(x|z) - logP(z)$ inside the expectation can again be reformulated as a KL divergence. Therefore we arrive at equation (3) and luckily, $ D_{KL}(Q(z|x)||P(z))$ is calculatable because $ P(z)$ is the gaussion $ N(0, I)$ we assigned before and $ Q(z|x)$ can be easily assigned as well. We simply make the encoder network predict the indicators: the mean $ \mu_{x}$ and the covariance $ \sigma_{x}$, appoximating $ Q(z|x)$ with a Guassian distribution determined by $ x$.

## The training objective

$ E_{z \sim Q}[logP(x|z)] - D_{KL}(Q(z|x)||P(z))$ is a lower bound of the final target $ P(x)$. In order to optimize $ P(x)$, we need to maximize $ E_{z \sim Q}[logP(x|z)]$ and minimize $ D_{KL}(Q(z|x)||P(z))$.

Because $ x$ is sampled from the guassain $ N(f(z;\theta), \sigma)$, we can optimize $ E_{z \sim Q}[logP(x|z)]$ by minimizing the reconstruction loss $ (x - f(z;\theta))^2$.

On the other hand, the KL divergence between two guassian can be calculated as follows:

$$
D(N(\mu0,\sigma0)||N(\mu1,\sigma1)) = \frac{1}{2}(tr(\sigma1^{-1}\sigma0)+(\mu1-\mu0)^T\sigma1^{-1}(\mu1-\mu0)-k+log\frac{det(\sigma1)}{det(\sigma0)})
$$

For a more detailed explanation and insights into the calculation of KL divergence between two Gaussian distributions, you can refer to the following link: [KL Divergence between 2 Gaussian Distributions (mr-easy.github.io)](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)

Now that $ P(z)$ equals to $ N(0, I)$, thus $ D_{KL}(Q(z|x)||P(z)) = \frac{1}{2}(tr(\sigma(x))+<\mu(x),\mu(x)>-k-logdet(\sigma(x)))$.

Combining the two terms, and we get the training objective to minimize:

$$
Loss(x) = (x - f(z;\theta))^2 + \frac{1}{2}(tr(\sigma(x))+<\mu(x),\mu(x)>-k-logdet(\sigma(x)))
$$

## Reparameterization Trick

![img]()

Note that when training, we input an image $ x$, and it passes through the encoder network $ g(;\theta^*)$ and put out $ \mu(x)$ and $ \sigma(x)$. Then we need to sample a $ z$ from the gaussion $ N(\mu(x),\theta(x))$. The trouble is this step of sampling is purly random and can't be calculated the gradient. Making it impossible for neural network to update its parameters.

Therefore we put forward the **Reparameterization Trick**. The idea is simple $ N(\mu(x),\sigma(x))=\mu(x)+\sigma(x)N(0,I)$. Thus during for each forward process, we sample a $ \epsilon\sim N(0,I) $ as an additional input, and reparameterize $ f(x;\theta)$ into $ f(x, \epsilon; \theta)$.

## Unconditional Generation

Once our VAE model is trained, we can employ it for unconditional image generation. The most common approach involves sampling a latent vector, $z$, from a standard Gaussian distribution $N(0, I)$ and passing this $z$ through the decoder network. The output, $f(z;\theta)$, represents a newly generated image.

Another valuable use of the VAE is generating images that resemble a given input image. This is accomplished by inputting the target image, denoted as $x$, into the encoder, which yields the mean latent vector $\mu(x)$. By adding a small disturbance, $\epsilon$, to $\mu(x)$, we obtain the latent vector $z = \mu(x) + \epsilon$. Subsequently, we follow the same procedure as mentioned above to generate an image that shares characteristics with the input image.

### Reference

[Variational Autoencoder(VAE). As a generative model, the basic idea… | by Roger Yong | Geek Culture | Medium](https://medium.com/geekculture/variational-autoencoder-vae-9b8ce5475f68)

[[1606.05908] Tutorial on Variational Autoencoders (arxiv.org)](https://arxiv.org/abs/1606.05908)

[Kullback–Leibler divergence - Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

[lec12_vae.pdf (illinois.edu)](https://slazebni.cs.illinois.edu/spring17/lec12_vae.pdf)

[KL Divergence between 2 Gaussian Distributions (mr-easy.github.io)](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)
