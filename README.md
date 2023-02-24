# Diffusion model in web browser


[Demo Site](https://wangjia184.github.io/diffusion_model/)


## 1. Diffusion Model Introduction

![](https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion_figure.png)

* $q$ - a fixed (or predefined) **forward** diffusion process of adding Gaussian noise to an image gradually, until ending up with pure noise
* $p_θ$ - a learned **reverse** denoising diffusion process, where a neural network is trained to gradually denoise an image starting from pure noise, until ending up with an actual image.

Both the forward and reverse process indexed by $t$ happen for some number of finite time steps $T$ (the DDPM authors use $T$=1000). You start with $t=0$ where you sample a real image $x_0$ from your data distribution, and the forward process samples some noise from a Gaussian distribution at each time step $t$, which is added to the image of the previous time step. Given a sufficiently large $T$ and a well behaved schedule for adding noise at each time step, you end up with what is called an *isotropic Gaussian distribution* at $t=T$ via a gradual process




## 2. Forward Process $q$

$$ x_0 \overset{q(x_1 | x_0)}{\rightarrow} x_1 \overset{q(x_2 | x_1)}{\rightarrow} x_2 \rightarrow \dots  \rightarrow x_{T-1} \overset{q(x_{t} | x_{t-1})}{\rightarrow} x_T $$


This process is a markov chain, $x_t$ only depends on $x_{t-1}$. $q(x_{t} | x_{t-1})$ adds Gaussian noise at each time step $t$, according to a known variance schedule $β_{t}$ 

$$ x_t = \sqrt{1-β_t}x_{t-1} + \sqrt{β_t}ϵ_{t} $$

* $β_t$ is not constant at each time step $t$. In fact one defines a so-called "variance schedule", which can be linear, quadratic, cosine, etc. 

$$ 0 < β_1 < β_2 < β_3 < \dots < β_T < 1 $$

* $ϵ_{t}$ Gaussian noise, sampled from standard normal distribution.





$$ x_t = \sqrt{1-β_t}x_{t-1} + \sqrt{β_t}ϵ_{t} $$

Define $a_t = 1 - β_t$

$$ x_t = \sqrt{a_{t}}x_{t-1} +  \sqrt{1-a_t} ϵ_{t} $$

### 2.1 Consider the relationship between $x_t$ and $x_{t-2}$

$$ x_{t-1} = \sqrt{a_{t-1}}x_{t-2} +  \sqrt{1-a_{t-1}} ϵ_{t-1}$$ 

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}} (\sqrt{a_{t-1}}x_{t-2} +  \sqrt{1-a_{t-1}} ϵ_{t-1}) +  \sqrt{1-a_t} ϵ_t $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{a_{t}(1-a_{t-1})} ϵ_{t-1} +  \sqrt{1-a_t} ϵ_t $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{a_{t}(1-a_{t-1}) + 1-a_t} * ϵ^{\prime} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{1-a_{t}a_{t-1}} * ϵ^{\prime} $$

### 2.2 Consider the relationship between $x_t$ and $x_{t-3}$

$$ x_{t-2} = \sqrt{a_{t-2}}x_{t-3} +  \sqrt{1-a_{t-2}} ϵ_{t-2} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}(\sqrt{a_{t-2}}x_{t-3} +  \sqrt{1-a_{t-2}} ϵ_{t-2}) +  \sqrt{1-a_{t}a_{t-1}}* ϵ^{\prime} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{a_{t}a_{t-1}(1-a_{t-2})} ϵ_{t-2} +  \sqrt{1-a_{t}a_{t-1}}* ϵ^{\prime} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{a_{t}a_{t-1}-a_{t}a_{t-1}a_{t-2}} ϵ_{t-2} +  \sqrt{1-a_{t}a_{t-1}}* ϵ^{\prime} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{(a_{t}a_{t-1}-a_{t}a_{t-1}a_{t-2}) + 1-a_{t}a_{t-1}} * ϵ^{\prime\prime} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}} * ϵ^{\prime\prime} $$

### 2.3 Consider the relationship between $x_t$ and $x_0$

* $x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{1-a_{t}a_{t-1}}*ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}}*ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}a_{t-3}}x_{t-4} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}a_{t-3}}*ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}x_{0} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}*ϵ$

$$\bar{a}_t=\prod_{s=1}^{t}a_{s}$$

$$x_{t} = \sqrt{\bar{a}_t}x_0+ \sqrt{1-\bar{a}_t}*ϵ , ϵ \in N(0,I) $$


