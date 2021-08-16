# AdaVC-for-optimizing-debt-collection

AdaVC is an adaptive voice conversion framework designed to improve debt collection performance. On the basis of fully learning the prosody information of the collectors, it converts the prosody of the bad collectors into the prosody of the good collectors, thereby realizing the improvement of the collection performance of the bad collectors.

## Methodology

The overview framework of AdaVC

![image](https://github.com/AdaVC-lab/AdaVC2021/blob/main/AdaVC-Overview.png)

The design of AdaVC conversion module

![image](https://github.com/AdaVC-lab/AdaVC2021/blob/main/conversion_module_flowchart.png)

The design motivation of AdaVC forge module

![image](https://github.com/AdaVC-lab/AdaVC2021/blob/main/forge_function_with_arrow.png)

If you find this work useful and use it in your research, please consider citing our paper.

## Audio demo

## Dependencies

## To training

The training process of AdaVC consists of two parts, the first is to train the timbre constraint, the prosody constraint and the discriminator, and then further train the reconstruction loss of mel.

Please use the scripts to prepare your own data for training.

- 1.Extract spectrogram and f0(conversion module): python data_loader.py

- 2.Run the training scripts(conversion module): python train.py

- 3.Run the forge scripts(forge module): python forge_demo.py
