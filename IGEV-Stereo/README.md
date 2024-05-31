## Environment
* NVIDIA RTX 4090
* Python 3.8
* Pytorch 1.13.1

### Create a virtual environment and activate it.

```
conda create -n IGEV_Stereo python=3.8
conda activate IGEV_Stereo
```
### Dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4
pip install torch-pruning==1.3.7
```

## Run
### Training original model
```
python train_stereo.py --logdir ./checkpoints/kitti --restore_ckpt ./pretrained_models/sceneflow.pth --train_datasets kitti
```
### Traing with pruning
```
python train_pruning_stereo.py --logdir ./checkpoints/pruned --restore_ckpt ./pretrained_models/original.pth --train_datasets kitti --amounts 0.2 0.1
```

