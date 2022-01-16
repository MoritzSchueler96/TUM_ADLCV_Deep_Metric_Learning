# About

This repository the official PyTorch implementation
of `Learning Intra-Batch Connections for Deep Metric Learning`. The config files contain the same parameters as used in the paper.

We use torch 1.10.0 and torchvision 0.11.1. While the training and inference should
be able to be done correctly with the newer versions of the libraries, be aware
that at times the network trained and tested using versions might diverge or reach lower
results. We provide a `env.yaml` file as well as installation instructions with a requirements.txt file (for model agnosticity) to create a corresponding conda environment.

We also support mixed-precision training via Nvidia Apex and describe how to use it in usage.

As in the paper we support training on 4 datasets: CUB-200-2011, CARS 196, Stanford Online Products and In-Shop datasets.

The majority of experiments are done using ResNet50. We
provide support for the entire family of ResNet and DenseNet as well as BN-Inception.

# Set up


1. Clone and enter this repository:

        git clone https://github.com/dvl-tum/intra_batch.git

        cd intra_batch

2. Create a Conda environment for this project:
To set up a conda environment containing all used packages, please fist install anaconda or miniconda and then install the environment using the  requirements.txt file:

   1.      conda create -n intra_batch_dml python=3.7
   2.      conda activate intra_batch_dml
   3.      pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   4.      pip install -r requirements.txt
   5.      cd libs/pytorch_geometric
           pip install -e .
   6.  (Optional) If you want to use Apex, please follow the installation instructions on https://github.com/NVIDIA/apex

        If you use Google Cloud Platform use the Python only version

3. If you want to use Google Cloud Platform:
   1.   Create a Deep Learning VM 
        (recommended: Tesla T4, 8core CPU with 10GB RAM, Ubuntu 20.04 with Pytorch 1.10 and Cuda 11.1 preinstalled and 100GB SSD storage -> costs ~1$/h, training time ~4h for 70 epochs)
   2.   Install Nvidia Driver
         prompted when connected via SSH (may not work at first, if so please run "sudo dpkg --configure -a" and re-login in shell and try again)
   3.   Use installations instruction from above
   4.   If you want to use Apex, use the python only installation:

            pip install -v --disable-pip-version-check --no-cache-dir ./

4. Download datasets:
Make a data directory by typing 

        mkdir data
    Then download the datasets using the following links and unzip them in the data directory:
    * Cars196: https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/CARS.zip
        ```bash
        wget https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/CARS.zip
        ```
    * CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    * Stanford Online Products: https://cvgl.stanford.edu/projects/lifted_struct/
        ```bash
        wget ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
        ```
    * In-Shop: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

    We also provide a parser for Stanford Online Products and In-Shop datastes. You can find dem in the `dataset/` directory. The datasets are expected to be structured as 
    `dataset/images/class/`, where dataset is either CUB-200-2011, CARS, Stanford_Online_Products or In_shop and class are the classes of a given dataset. Example for CUB-200-2011: 

            CUB_200_2011/images/001
            CUB_200_2011/images/002
            CUB_200_2011/images/003
            ...
            CUB_200_2011/images/200


4. Download our models: Please download the pretrained weights by using

        wget https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/best_weights.zip

    and unzip them in the home directory. This will give you a folder best_weights with all trained models.

# Usage
You can find config files for training and testing on each of the datasets in the `config/` directory. For training and testing, you will have to input which one you want to use (see below). You will only be able to adapt some basic variables over the command line. For all others please refer to the yaml file directly.

## Testing
To test to networks choose one of the config files for testing, e.g., `config_cars_test.yaml` to evaluate the performance on Cars196 and run in the root directory:

    python train.py --config_path <path to config> --dataset_path <path to dataset> 

In example:

    python train.py --config_path config/config_cars_test.yaml --dataset_path data/CARS

The default dataset path is data.

## Training
To train a network choose one of the config files for training like `config_cars_train.yaml` to train on Cars196 and run:

    python train.py --config_path <path to config> --dataset_path <path to dataset> --net_type <net type you want to use>

In example:

    python train.py --config_path config/config_cars_train.yaml --dataset_path data/CARS --net_type resnet50

Again, if you don't specify anything, the default setting will be used. For the net type you have the following options:

`resnet18, resnet32, resnet50, resnet101, resnet152, densenet121, densenet161, densenet16, densenet201, bn_inception`

If you want to use apex add `--is_apex 1` to the command.


# Results
|               | R@1   | R@2   | R@4   | R@8   | NMI   |
| ------------- |:------|------:| -----:|------:|------:|
| CUB-200-2011  | 70.3  | 80.3  | 87.6  | 92.7  | 73.2  |
| Cars196       | 88.1  | 93.3  | 96.2  | 98.2  | 74.8  |

|                            | R@1   | R@10  | R@100 | NMI   |
| -------------------------- |:------|------:| -----:|------:|
| Stanford Online Products   | 81.4  | 91.3  | 95.9  | 92.6  |

|               | R@1   | R@10  | R@20  | R@40  |
| ------------- |:------|------:| -----:|------:|
| In-Shop       | 92.8  | 98.5  | 99.1  | 99.2  |

# Citation

If you find this code useful, please consider citing the following paper:

```
@inproceedings{DBLP:conf/icml/SeidenschwarzEL21,
  author    = {Jenny Seidenschwarz and
               Ismail Elezi and
               Laura Leal{-}Taix{\'{e}}},
  title     = {Learning Intra-Batch Connections for Deep Metric Learning},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {9410--9421},
  publisher = {{PMLR}},
  year      = {2021},
}
```
