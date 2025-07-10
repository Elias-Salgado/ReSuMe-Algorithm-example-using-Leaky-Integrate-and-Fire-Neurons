We denote firing times of neuron i by $t^{(f)}_{i}$	where $f =1,2,...$ is the label of the spike. Formally, we may denote the spike train of a neuron $i$ as the sequence of firing times
```math
		S_{i}\left(t\right)=\sum_{f}^{}\delta\left(t-t_{i}^{(f)}\right)
```
<br />

where  $\delta\left(x\right)$ is the Dirac function with $\delta\left(x\right)=0$ for $x\neq 0$ and $\int_{-\infty}^{\infty}\delta\left(x\right) = 1$. Spikes are thus reduced to points in time.<br />
The kernel $`\epsilon_{i,j}\left(t-\hat{t}_{i},s\right)`$ as a function of $s = t-t_{j}^{(f)}$ can be interpreted as the time course of a postsynaptic potential evoked by the firing of a presynaptic neuron $j$ at time $t_{j}^{(f)}$.
<br />
The leaky integrate and fire potential can be described as:
	
```math
		u(t) = \sum_{j}w_{i,j}\sum_{f}\epsilon_{i,j}\left(t-\hat{t}_{i},s\right)
```
where the kernel function $\epsilon_{i,j}\left(t-\hat{t}_{i},s\right)$ can be mapped as follows:
	
```math
		\epsilon(s,t) = \frac{exp\left(-\frac{max(s,0)}{\tau_{s}}\right)}{1-\frac{\tau_{s}}{\tau_{m}}}\left[exp\left(-\frac{min(s,t)}{\tau_{m}}\right)-exp\left(-\frac{min(s,t)}{\tau_{s}}\right)\right]\Theta(s)\Theta(t)
```

<p align="center">
  <img width="1000" src="https://github.com/user-attachments/assets/72ed340b-adba-421a-9eba-a2810cfbb42d">
</p>
<p align="center">
    <em>Fig. 1: Second order Leaky Integrate and Fire synaptic potential.</em>
</p>

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/1aece3b6-05e2-4ad9-a84e-55172d86e200">
</p>

<p align="center">
    <em>Fig. 2: Training scheme for signle neuron, second order Leaky Integrate and Fire synapses.</em>
</p>

<p align="center">
  <img width="800" src="https://github.com/user-attachments/assets/533a2e49-501b-4120-a52f-9b8bdf1d2b13">
</p>

<p align="center">
    <em>Fig. 3: Updating process of RESUME algorithm.</em>
</p>

<p align="center">
  <img width="1000" src="https://github.com/user-attachments/assets/c3b18fa9-ec4a-4ef7-9020-9c457cdd39b6">
</p>
<p align="center">
    <em>Fig. 4: Output spikes for reference and post-train models.</em>
</p>
