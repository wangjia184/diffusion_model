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

$$ x_t = \sqrt{1-β_t}x_{t-1} + \sqrt{β_t}\times ϵ_{t} $$

* $β_t$ is not constant at each time step $t$. In fact one defines a so-called "variance schedule", which can be linear, quadratic, cosine, etc. 

$$ 0 < β_1 < β_2 < β_3 < \dots < β_T < 1 $$

* $ϵ_{t}$ Gaussian noise, sampled from standard normal distribution.





$$ x_t = \sqrt{1-β_t}x_{t-1} + \sqrt{β_t} \times ϵ_{t} $$

Define $a_t = 1 - β_t$

$$ x_t = \sqrt{a_{t}}x_{t-1} +  \sqrt{1-a_t} \times ϵ_{t} $$

### 2.1 Relationship between $x_t$ and $x_{t-2}$

$$ x_{t-1} = \sqrt{a_{t-1}}x_{t-2} +  \sqrt{1-a_{t-1}} \times ϵ_{t-1}$$ 

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}} (\sqrt{a_{t-1}}x_{t-2} +  \sqrt{1-a_{t-1}} ϵ_{t-1}) +  \sqrt{1-a_t} \times ϵ_t $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{a_{t}(1-a_{t-1})} ϵ_{t-1} +  \sqrt{1-a_t} \times ϵ_t $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{a_{t}(1-a_{t-1}) + 1-a_t} \times ϵ $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{1-a_{t}a_{t-1}} \times ϵ $$

### 2.2 Relationship between $x_t$ and $x_{t-3}$

$$ x_{t-2} = \sqrt{a_{t-2}}x_{t-3} +  \sqrt{1-a_{t-2}} \times ϵ_{t-2} $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}}(\sqrt{a_{t-2}}x_{t-3} +  \sqrt{1-a_{t-2}} ϵ_{t-2}) +  \sqrt{1-a_{t}a_{t-1}}\times ϵ $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{a_{t}a_{t-1}(1-a_{t-2})} ϵ_{t-2} +  \sqrt{1-a_{t}a_{t-1}}\times ϵ $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{a_{t}a_{t-1}-a_{t}a_{t-1}a_{t-2}} ϵ_{t-2} +  \sqrt{1-a_{t}a_{t-1}}\times ϵ $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3}  +  \sqrt{(a_{t}a_{t-1}-a_{t}a_{t-1}a_{t-2}) + 1-a_{t}a_{t-1}} \times ϵ $$

$$ \Downarrow  $$

$$ x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}} \times ϵ $$

### 2.3 Relationship between $x_t$ and $x_0$

* $x_t = \sqrt{a_{t}a_{t-1}}x_{t-2} +  \sqrt{1-a_{t}a_{t-1}}\times ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}x_{t-3} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}}\times ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}a_{t-3}}x_{t-4} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}a_{t-3}}\times ϵ$
* $x_t = \sqrt{a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}x_{0} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}\times ϵ$

$$\bar{a}_t=\prod_{s=1}^{t}a_{s}$$

$$x_{t} = \sqrt{\bar{a}_t}x_0+ \sqrt{1-\bar{a}_t}\times ϵ , ϵ \in N(0,I) $$

$$ \Downarrow  $$

$$ q(x_{t}|x_{0}) = \frac{1}{\sqrt{2\pi } \sqrt{1-\bar{a}_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}   \right ) } $$

# 3.Reverse Process $p$

$$ P(A|B) = \frac{ P(B|A)\times P(A) }{P(B)} $$

$$ p(x_{t-1}|x_{t}) = \frac{ q(x_{t}|x_{t-1})\times q(x_{t-1}|x_0)}{q(x_{t}|x_0)} $$

<table>
  <tbody>
    <tr>
      <td>
         $$x_{t} = \sqrt{a_t}x_{t-1}+\sqrt{1-a_t}\times ϵ$$
      </td>
      <td>
        ~
      </td>
      <td>
        $N(\sqrt{a_t}x_{t-1}, 1-a_{t})$
      </td>
    </tr>
    <tr>
      <td>
        $$x_{t-1} = \sqrt{\bar{a}_{t-1}}x_0+ \sqrt{1-\bar{a}_{t-1}}\times ϵ$$
      </td>
      <td>
        ~
      </td>
      <td>
        $N( \sqrt{\bar{a}_{t-1}}x_0, 1-\bar{a}_{t-1})$
      </td>
    </tr>
    <tr>
      <td>
        $$x_{t} = \sqrt{\bar{a}_{t}}x_0+ \sqrt{1-\bar{a}_{t}}\times ϵ$$
      </td>
      <td>
        ~
      </td>
      <td>
        $N( \sqrt{\bar{a}_{t}}x_0, 1-\bar{a}_{t})$
      </td>
    </tr>
  </tbody>
  
</table>


$$ q(x_{t}|x_{t-1}) = \frac{1}{\sqrt{2\pi } \sqrt{1-a_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{a_t}x_{t-1})^2}{1-a_{t}}   \right ) } $$

$$ q(x_{t-1}|x_{0}) = \frac{1}{\sqrt{2\pi } \sqrt{1-\bar{a}_{t-1}}} e^{\left (  -\frac{1}{2}\frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}}   \right ) } $$

$$ q(x_{t}|x_{0}) = \frac{1}{\sqrt{2\pi } \sqrt{1-\bar{a}_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}   \right ) } $$





$$ \frac{ q(x_{t}|x_{t-1})\times q(x_{t-1}|x_0)}{q(x_{t}|x_0)} = \left [
  \frac{1}{\sqrt{2\pi} \sqrt{1-a_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{a_t}x_{t-1})^2}{1-a_{t}}   \right ) } 
\right ] * 
\left [ 
\frac{1}{\sqrt{2\pi} \sqrt{1-\bar{a}_{t-1}}} e^{\left (  -\frac{1}{2}\frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}}   \right ) }  
\right ] \div
\left [ 
  \frac{1}{\sqrt{2\pi} \sqrt{1-\bar{a}_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}   \right ) }
\right ]  $$

$$ \Downarrow  $$

$$ \frac{\sqrt{2\pi} \sqrt{1-\bar{a}_{t}}}{\sqrt{2\pi} \sqrt{1-a_{t}} \sqrt{2\pi} \sqrt{1-\bar{a}_{t-1}} }
e^{\left [  -\frac{1}{2}
\left (
 \frac{(x_{t}-\sqrt{a_t}x_{t-1})^2}{1-a_{t}} +
 \frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}} -
 \frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}
 \right )
    \right ] } $$

$$ \Downarrow  $$

$$ \frac{1}{\sqrt{2\pi} \left ( \frac{ \sqrt{1-a_t} \sqrt{1-\bar{a}_{t-1}} } {\sqrt{1-\bar{a}_{t}}} \right ) }
exp{\left [  -\frac{1}{2}
\left (
 \frac{(x_{t}-\sqrt{a_t}x_{t-1})^2}{1-a_t} +
 \frac{(x_{t-1}-\sqrt{\bar{a}_{t-1}}x_0)^2}{1-\bar{a}_{t-1}} -
 \frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}
 \right )
    \right ] } $$

$$ \Downarrow  $$

$$ \frac{1}{\sqrt{2\pi} \left ( \frac{ \sqrt{1-a_t} \sqrt{1-\bar{a}_{t-1}} } {\sqrt{1-\bar{a}_{t}}} \right ) }
exp \left[  -\frac{1}{2}
\left (
 \frac{
   x_{t}^2-2\sqrt{a_t}x_{t}x_{t-1}+{a_t}x_{t-1}^2
 }{1-a_t} +
 \frac{
   x_{t-1}^2-2\sqrt{\bar{a}_{t-1}}x_0x_{t-1}+\bar{a}_{t-1}x_0^2
  }{1-\bar{a}_{t-1}} -
 \frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}
\right)
\right] $$






$$ \Downarrow  $$

$$ \frac{1}{\sqrt{2\pi} \left ( \frac{ \sqrt{1-a_t} \sqrt{1-\bar{a}_{t-1}} } {\sqrt{1-\bar{a}_{t}}} \right ) }  
exp \left[
-\frac{1}{2}
\frac{
  \left(
    x_{t-1} - \left(
      \frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t
      +
      \frac{\sqrt{\bar{a}_{t-1}}(1-a_t)}{1-\bar{a}_t}x_0
      \right)
  \right) ^2
} {   \left( \frac{ \sqrt{1-a_t} \sqrt{1-\bar{a}_{t-1}} } {\sqrt{1-\bar{a}_{t}}} \right)^2 }
\right] $$


$$ \Downarrow  $$

$$ ~ N\left( 
      \frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t
      +
      \frac{\sqrt{\bar{a}_{t-1}}(1-a_t)}{1-\bar{a}_t}x_0 ,
      \left( \frac{ \sqrt{1-a_t} \sqrt{1-\bar{a}_{t-1}} } {\sqrt{1-\bar{a}_{t}}} \right)^2
 \right) $$







