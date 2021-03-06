# iCaRL: Incremental Classifier and Representation Learning
Incremental learning is a machine learning paradigm in which a model is trained continuously on new input data whenever it is available, extending the existing model's knowledge. In our dissertation, we experiment on three incremental learning methods, Finetuning, Learning without Forgetting and iCaRL. Our study then focuses more in-depth on the iCaRL framework, which makes use of the techniques of knowledge distillation and prototype rehearsal to preserve past knowledge acquired by a deep convolutional neural network, while simultaneously training on new classes. We then perform an ablation study, experimenting with different combinations of distillation and classification losses and introducing new classifiers. By inspecting the behavior of the newly introduced elements, we seek to better understand how the model benefits from each of its individual components and what its possible weaknesses are.
We subsequently propose some variations of the original iCaRL algorithm that attempt to tackle such issues, and we verify their effectiveness. For a fair comparison, we perform our tests on CIFAR-100, as used in the original
iCaRL paper. Eventually, our extensions of iCaRL lead to a model achieving a better accuracy than iCaRL on CIFAR-100.

The repository contains the code for Incremental Learning project. In the master branch, the files contains:

  - FineTuning -> Fine Tuning version; 
  - LWF -> LWF version; 
  - ICaRLMain ICaRL;

New proposal and Ablation studies are available in different branches:

  - Loss---* branches with different proposals of losses.
  - Classifier---* are branches with different proposals of classifiers.
  - Proposals---* are branches with different approaches followed in the study.
