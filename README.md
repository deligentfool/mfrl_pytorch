# Pytorch Version for Mean Field Multi-Agent Reinforcement Learning
Pytorch implementation of MF-Q and MF-AC in the paper [Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1802.05438.pdf).

The original code can be found in [mlii/mfrl](https://github.com/mlii/mfrl).

Please uncomment the following two lines of code in `base.py` if the algorithm occasionally fails to converge.
```python
  #distribution = torch.distributions.Categorical(predict) 
  #actions = distribution.sample().detach().cpu().numpy()
```

## Example
![image](https://github.com/deligentfool/mfrl_pytorch/blob/master/replay.gif)
