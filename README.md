# Generation of multilingual text

This is PyTorch implementation of a VAE for multilingual text generation.

A detailed report of our work can be found in the file "rapport_MAP511".

## Requirements

* Python >= 3.6
* PyTorch >= 1.0
* pip install editdistance

## Data

3 datasets are presented in the folder "datasets". 

The "tatoeba_data" dataset can be used to train a VAE using the method described here : https://github.com/bohanli/vae-pretraining-encoder
The other two are processed to be used with our method.

One can use another dataset and preprocees it with the file test.py

## Usage

Train a AE first
```
python text_beta2.py \
    --dataset tatoeba2spm \
    --beta 0 \
    --lr 0.5
```

Train VAE with our method
```
ae_exp_dir=exp_tatoeba2spm_beta/tatoeba2spm_lr0.5_beta0.0_drop0.5_
python text_anneal_fb2.py \
    --dataset tatoeba2spm \
    --load_path ${ae_exp_dir}/model.pt \
    --reset_dec \
    --kl_start 0 \
    --warm_up 10 \
    --target_kl 8 \
    --fb 2 \
    --lr 0.5
```

Create homotopies
```
vae_exp_dir=exp_tatoeba2spm_load/tatoeba2spm_warm10_kls0.0_fbdim_tr8.0
python homotopie.py \
    --dataset tatoeba2spm \
    --load_path ${vae_exp_dir}/model.pt \
    --fb 2 \
    --lr 0.5
```

Logs, models and samples would be saved into folder `exp`.


## Acknowledgements

A large portion of this repo is borrowed from https://github.com/bohanli/vae-pretraining-encoder

