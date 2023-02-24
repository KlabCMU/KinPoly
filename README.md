# Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation

[[paper]](https://arxiv.org/abs/2106.05969) [[website]](https://zhengyiluo.github.io/projects/kin_poly/) [[Video]](https://www.youtube.com/watch?v=yEiK9K1N-zw)

<div float="center">
  <img src="assets/gif/teaser.gif" />
</div>

## News 🚩
[February 24, 2023] Updating some model loading; please rerun  `bash download_data.sh` to get the newest models 🙏🏻

[Ocotober 20, 2022] Updating to runnable kinemaitc policy 

[Ocotober 20, 2021] Code release!

## Introduction

In this project, we demonstrate the ability to estimate 3D human pose and human-object interactions from egocentric videos. This code base contains all the necessary files to train and reproduce the results reported in our paper, and contain configuration files and hyperparameters used in our experiments. Some training data (namely, [AMASS](https://amass.is.tue.mpg.de/)) and external library ([Mujoco](http://www.mujoco.org/)) may require additional licence to obtain, and this codebase contains data processing scripts to process these data once obtained. 


## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:
```
conda create -n kin_poly python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. Download and setup mujoco: [Mujoco](http://www.mujoco.org/)
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

3. The rest of the dependencies can be found in ```requirements.txt```. 

```
pip install -r requirements.txt
```

## Datasets and trained models

The datasets we use for training and evaluating our method can be found here:

[[Real-world dataset](https://drive.google.com/drive/folders/1BBjPmjrm-FZLMw24Gsbl4CsodGgfsptY?usp=sharing)][[MoCap dataset](https://drive.google.com/drive/folders/1Mw1LQBNfor8a7Diw3eHLO--ZnREw57kB?usp=sharing)]

The folders contain the a data file that contains the pre-computed object pose and camera trajectory; another data file contains the pre-computed image features; a meta file is also included for loading the respective datasets.

To download the Mocap dataset, real-world dataset, and trained models, run the following script: 

```
bash download_data.sh
```

## Important files

* ```kin_poly/models/traj_ar_smpl_net.py```:  definition of our kinematic model's network.
* ```kin_poly/models/policy_ar.py```:  wrapper around our kinematic model to form the kinematic policy.
* ```kin_poly/envs/humanoid_ar_v1.py```: main Mujoco environment for training and evaluating our kinematic policy.
* ```scripts/eval_pose_all.py```: evaluation code that computes all metrics reported in our paper from a pickled result file. 
* ```config/kin_poly.yml```: the configuration file used to train our kinematic policy.
* ```uhc/cfg/uhc.yml```: the configuration file used to train our universal humanoid controller.
* ```assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml```: the simulation configuration used in our experiments. It contains the definition for the humanoid and the objects (chair, box, table, etc.) for Mujoco. 

## Training

To train our dynamics-regulated kinematic policy, use the command:

```
python scripts/train_ar_policy.py --cfg kin_poly  --num_threads 35 
```

To train our kinematic policy using only supervised learning, use the command:

```
python scripts/exp_arnet_all.py --cfg kin_poly  
```

To train our universal humanoid controller, use the command:

```
python scripts/train_uhc.py.py --cfg uhc --num_threads 35
```

## Evaluation

To evaluate our dynamics-regulated kinematic policy, run:
```
python scripts/eval_ar_policy.py --cfg kin_poly --iter 750  # Mocal data
python scripts/eval_ar_policy.py --cfg kin_poly --iter 750  --wild # Real-world data
```

To evalutes our kinematic policy using only supervised learning, run:
```
python scripts/exp_arnet_all.py --cfg kin_poly  --test --iter 1000
```

To compute metrics, run:
```
python scripts/eval_pose_all --cfg kin_poly --algo kin_poly --iter 750
```

To evaluate our universal humanoid controller, run:
```
python scripts/eval_uhc.py --cfg uhc --iter 10000
```

## Relationship to the main [uhc](https://github.com/KlabCMU/UniversalHumanoidControl) repository
This repository is self-contained and houses an eariler version of the universal humanoid controller (one that only supports the average neutral SMPL human). For support of all SMPL human models, please refer to the main UHC repository.


## Citation
If you find our work useful in your research, please cite our paper [kin_poly](https://zhengyiluo.github.io/projects/kin_poly/):
```
@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

``` 

Related work: 
```
@inproceedings{yuan2020residual,
    title={Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis},
    author={Yuan, Ye and Kitani, Kris},
    booktitle={Advances in Neural Information Processing Systems},
    year={2020}
}

@inproceedings{yuan2019ego,
    title={Ego-Pose Estimation and Forecasting as Real-Time PD Control},
    author={Yuan, Ye and Kitani, Kris},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2019},
    pages={10082--10092}
}

```

## References
This repository is based on the following repositories:
* Part of the UHC code is from: [rfc](https://github.com/Khrylx/RFC)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
* Feature extractors are from: [SPIN](https://github.com/nkolot/SPIN)
* NN modules are from (khrylib): [DLOW](https://github.com/Khrylx/DLow)
