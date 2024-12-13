
# 597-fa24-project
## Replicating - ExpansionNet v2: Exploiting Multiple Sequence Lengths in Fast End to End Training for Image Captioning

Implementation code for "[Exploiting Multiple Sequence Lengths in Fast End to End Training for Image Captioning](https://www.computer.org/csdl/proceedings-article/bigdata/2023/10386812/1TUPyooQsnu)" [ [BigData2023](https://www.computer.org/csdl/proceedings-article/bigdata/2023/10386812/1TUPyooQsnu) ]
[ [Arxiv](https://arxiv.org/abs/2208.06551) ], previously entitled as "ExpansionNet v2: Block Static Expansion
in fast end to end training for Image Captioning". <br>


## Training

#### Requirements

* python >= 3.7
* numpy
* Java 1.8.0
* torch
* torchvision
* h5py

Installing whatever version of `torch, torchvision, h5py, Pillow` fit your machine should
work in most cases.

One instance of requirements file can be found in `requirements.txt`, in case also TensorRT is needed
use `requirements_wTensorRT.txt`. However they represent one working instance, specific versions of each package 
might not be required. 

#### Data preparation

MS-COCO 2014 images can be downloaded [here](https://cocodataset.org/#download), 
the respective captions are uploaded in our online [drive](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing)
and the backbone can be found [here](https://github.com/microsoft/Swin-Transformer). All files, in particular
the `dataset_coco.json` file and the backbone are suggested to be moved in `github_ignore_materal/raw_data/` since commands provided
in the following steps assume these files are placed in that directory.


#### 1. Cross Entropy Training: Features generation

First we generate the features for the first training step:
```
cd ExpansionNet_v2_src
python data_generator.py \
    --save_model_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ \
    --captions_path ./github_ignore_material/raw_data/ &> output_file.txt &
```
Even if it's suggested not to do so, the `output_path` argument can be replaced with the desired destination (this would require
changing the argument `features_path` in the next commands as well). Since it's a pretty big
file (102GB), once the first training is completed, it will be automatically overwritten by
the remaining operations in case the default name is unchanged.

<b>TIPS:</b> if 100GB of memory is too much for your disk, add the option `--dtype fp16`
which saves arrays into FP16 so it requires only 50GB. It shouldn't change affect much the result.
By default, we keep FP32 for conformity with the experimental setup of the paper.


#### 2. Cross-Entropy Training: Partial Training

In this step the model is trained using the Cross Entropy loss and the features generated
in the previous step:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --seed 775533 --optim_type radam --sched_type custom_warmup_anneal  \
    --warmup 10000 --lr 2e-4 --anneal_coeff 0.8 --anneal_every_epoch 2 --enc_drop 0.3 \
    --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
    --batch_size 48 --num_accum 1 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --features_path ./github_ignore_material/raw_data/features.hdf5 --partial_load False \
    --print_every_iter 11807 --eval_every_iter 999999 \
    --reinforce False --num_epochs 8 &> output_file.txt &
```

#### 3. Cross-Entropy Training: End to End Training

The following command trains the entire network in the end to end mode. However,
one argument need to be changed according to the previous result, the
checkpoint name file. Weights are stored in the directory `github_ignore_materal/saves/`,
with the prefix `checkpoint_ ... _xe.pth` we will refer it as `phase2_checkpoint` below and in
the later step:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533   --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 3e-5 --anneal_coeff 0.55 --anneal_every_epoch 1 --enc_drop 0.3 \
    --dec_drop 0.3 --enc_input_drop 0.3 --dec_input_drop 0.3 --drop_other 0.3  \
    --batch_size 16 --num_accum 3 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [3]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ --partial_load True \
    --backbone_save_path ./github_ignore_material/raw_data/swin_large_patch4_window12_384_22k.pth \
    --body_save_path ./github_ignore_material/saves/phase2_checkpoint \
    --print_every_iter 15000 --eval_every_iter 999999 \
    --reinforce False --num_epochs 2 &> output_file.txt &
```
In case you are interested in the network's weights at the end of this stage, 
before moving to the self-critical learning, rename the checkpoint file from `checkpoint_ ... _xe.pth` into something 
else like `phase3_checkpoint` (make sure to change the prefix) otherwise it will 
be overwritten during step 5.

#### 4. CIDEr optimization: Features generation

This step generates the features for the reinforcement step:
```
python data_generator.py \
    --save_model_path ./github_ignore_material/saves/phase3_checkpoint \
    --output_path ./github_ignore_material/raw_data/features.hdf5 \
    --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ \
    --captions_path ./github_ignore_material/raw_data/ &> output_file.txt &
```

#### 5. CIDEr optimization: Partial Training

The following command performs the partial training using the self-critical learning:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533  --sched_type custom_warmup_anneal  \
    --warmup 1 --lr 1e-4 --anneal_coeff 0.8 --anneal_every_epoch 1 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end False --partial_load True \
    --features_path ./github_ignore_material/raw_data/features.hdf5 \
    --body_save_path ./github_ignore_material/saves/phase3_checkpoint.pth \
    --print_every_iter 4000 --eval_every_iter 99999 \
    --reinforce True --num_epochs 9 &> output_file.txt &
```
We refer to the last checkpoint produced in this step as `phase5_checkpoint`,
it should already achieve around 139.5 CIDEr-D on both Validaton and Test set, however
it can be still improved by a little margin with the following optional step.


#### 6. CIDEr optimization: End to End Training

This last step again train the model in an end to end fashion, however it is optional since it only slightly improves the performances:
```
python train.py --N_enc 3 --N_dec 3  \
    --model_dim 512 --optim_type radam --seed 775533 --sched_type custom_warmup_anneal  \
    --warmup 1 --anneal_coeff 1.0 --lr 2e-6 --enc_drop 0.1 \
    --dec_drop 0.1 --enc_input_drop 0.1 --dec_input_drop 0.1 --drop_other 0.1  \
    --batch_size 24 --num_accum 2 --num_gpus 1 --ddp_sync_port 11317 --eval_beam_sizes [5]  \
    --save_path ./github_ignore_material/saves/ --save_every_minutes 60 --how_many_checkpoints 1  \
    --is_end_to_end True --images_path ./github_ignore_material/raw_data/MS_COCO_2014/ --partial_load True \
    --backbone_save_path ./github_ignore_material/raw_data/phase3_checkpoint \
    --body_save_path ./github_ignore_material/saves/phase5_checkpoint \
    --print_every_iter 15000 --eval_every_iter 999999 \
    --reinforce True --num_epochs 1 &> output_file.txt &
```

## Evaluation

In this section we provide the evaluation scripts. We refer to the
last checkpoint as `phase6_checkpoint`. In case the previous training 
procedures have been skipped, 
weights of one of the ensemble's model can be found [here](https://drive.google.com/drive/folders/1bBMH4-Fw1LcQZmSzkMCqpEl0piIP88Y3?usp=sharing).
```
python test.py --N_enc 3 --N_dec 3 --model_dim 512 \
    --num_gpus 1 --eval_beam_sizes [5] --is_end_to_end True \
    --eval_parallel_batch_size 4 \
    --images_path ./github_ignore_material/raw_data/<your_coco_img_folder> \
    --save_model_path ./github_ignore_material/saves/phase6_checkpoint
```
The option `is_end_to_end` can be toggled according to the model's type. <br>
It might be required to give permissions to the file `./eval/get_stanford_models.sh` (e.g. `chmod a+x -R ./eval/` in Linux).


## Citation

If you find this repository useful, please consider citing the original authors:

```
@inproceedings{hu2023exploiting,
  title={Exploiting Multiple Sequence Lengths in Fast End to End Training for Image Captioning},
  author={Hu, Jia Cheng and Cavicchioli, Roberto and Capotondi, Alessandro},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={2173--2182},
  year={2023},
  organization={IEEE Computer Society}
}
```
