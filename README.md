# PolypSegDiff: Dynamic Multi-scaleConditional Diffusion Model for Polyp Segmentation



## Requirements

- python == 3.9
- cuda == 11.3

To install requirements:

```setup
pip install -r requirements.txt
```



## Train

```
python train.py --config config/psd_256x256.yaml --num_epoch=200 --batch_size=32 --gradient_accumulate_every=1
```

​    

## Sample

```
python sample.py \
  --config config/psd_256x256.yaml \
  --results_folder ${RESULT_SAVE_PATH} \
  --checkpoint ${CHECKPOINT_PATH} \
  --num_sample_steps 10 \
  --target_dataset POLYP \
  --time_ensemble
```

## Citation

```
@article{du2024polypsegdiff,
  title={PolypSegDiff: Dynamic Multi-scale Conditional Diffusion Model for Polyp Segmentation},
  author={Du, Xiaogang and Jiao, Yipeng and Lei, Tao and Zhang, Xuejun and Wang, Yingbo and Nandi, Asoke K},
  booktitle={International Conference on Pattern Recognition},
  pages={94--108},
  year={2024},
  organization={Springer}
}
```

```
@article{chen2023camodiffusion,
  title={CamoDiffusion: Camouflaged Object Detection via Conditional Diffusion Models},
  author={Chen, Zhongxi and Sun, Ke and Lin, Xianming and Ji, Rongrong},
  journal={arXiv preprint arXiv:2305.17932},
  year={2023}
}
```
