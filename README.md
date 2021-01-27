<h1 align="center"> Team Lynx - RESCON </h1>
<h3 align="center"> Original Repository of Team Lynx for ResCon 2021 <h3>

![RESCON](https://user-images.githubusercontent.com/70643852/105801454-27705e80-5fbf-11eb-97c7-9cc78d4b2636.png)
<h4 align="center"> Interactions between the Feature and Attention Masks of the Residual Attetntion Network [Image Referenced from the Paper] <h4>

## Basic Model we have implemeted

We have used the PyTorch-Lightning Framework to implement a Residual Attention Network which can be used for Image Classification. A Residual Attention Network is nothing but a, "Convolutional Neural Network using Attention Mechanism which can implement State-of-art feed forward network architecture", as mentioned in the abstract of the paper.

## Observations drawn from our implementation

| Dataset Used | Architecture implemented (Attention Type) | Optimiser Used | Image size | Training Loss | Test Loss | Framework Used |
|---|---|---|---|---|---|---|
| CIFAR-100 | Attention-92  | SGD | 32  | 1.26 | 1.58 | PyTorch-Lightning|
| CIFAR-10  | Attention-92  | SGD | 32  | 0.51 | 0.53 | PyTorch-Lightning|
| CIFAR-100 | Attention-56  | SGD | 224 | 1.42 | 1.80 | PyTorch-Lightning|
| CIFAR-10  | Attention-56  | SGD | 224 | 0.61 | 0.65 | PyTorch-Lightning|
| CIFAR-100 | Attention-92  | SGD | 224 | 2.95 | 2.90 | PyTorch-Lightning|
| CIFAR-10  | Attention-92  | SGD | 224 | 1.12 | 1.01 | PyTorch-Lightning|

## Further Improvements
We were able to implement only a few Res-Net architectures and that too only on 2 datasets because of the computional power and time required to run the model on our machines. Areas we are looking to improve on and work in the future
- [X] Implementing Attention-56 Architecture
- [X] Implementing Attention-92 Architecture
- [ ] Implementing Attention-128, Attention-156 Architecture
- [ ] Implementing the Paper using other Deep Learning Frameworks like Tensorflow

## Paper implemented
[Residual Attention Network for Image Classification By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang](https://github.com/AlKun25/ResCon/blob/Pranav/%5BPaper%5D%20Residual%20Attention%20Network%20for%20Image%20Classification.pdf)

## References
[ResidualAttentionNetwork-pytorch (GitHub)](https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch)
<br/>
[Residual Attention Network (GitHub)](https://github.com/fwang91/residual-attention-network)

### Citations

    @inproceedings{wang2017residual,
      title={Residual attention network for image classification},
      author={Wang, Fei and Jiang, Mengqing and Qian, Chen and Yang, Shuo and Li, Cheng and Zhang, Honggang and Wang, Xiaogang and Tang, Xiaoou},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={3156--3164},
      year={2017}
    }

## Contributors
[Harshit Aggarwal](https://github.com/harshitaggarwal01)

[Kunal Mundada](https://github.com/AlKun25)

[Pranav B](https://github.com/Pranav1007)
