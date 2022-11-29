
# [Towards Adversarial Attack on Vision-Language Pre-training Models](https://arxiv.org/abs/2206.09391)

This is the official PyTorch implement of the paper "[Towards Adversarial Attack on Vision-Language Pre-training Models](https://arxiv.org/abs/2206.09391)" at *ACM Multimedia 2022*. 

<!-- <img src="img.png" width=500> -->

## Update 29/11/2022
We released the fine-tuned checkpoints ([Baidu](https://pan.baidu.com/s/1hHkSBgv23rx0zSywBXwwWA?pwd=iqvp), password: iqvp) for VE task on ALBEF and TCL, which can be considered not only as an attacked model in this paper, but also useful for other ways.



## Requirements
- pytorch 1.10.2
- transformers 4.8.1
- timm 0.4.9
- bert_score 0.3.11


## Download
- Dataset json files for downstream tasks [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for ALBEF [[ALBEF github]](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for TCL [[TCL github]](https://github.com/uta-smile/TCL)


## Evaluation
|Adv|Instruction|
|---|---|
|0|No Attack|
|1|Attack Text|
|2|Attack Image|
|3|Attack Both (vanilla)|
|4|Co-Attack|

When attack unimodal embedding, using "--adv 4" and not using "--cls" will raise an expected error due to the different sequence length of image embedding and text embedding. 
### Image-Text Retrieval
Download MSCOCO or Flickr30k datasets from origin website.
```
# Attack Unimodal Embedding
python RetrievalEval.py --adv 4 --gpu 0 --cls \
--config configs/Retrieval_flickr.yaml \
--output_dir output/Retrieval_flickr \
--checkpoint [Finetuned checkpoint]

# Attack Multimodal Embedding
python RetrievalFusionEval.py ...

# Attack Clip Model
python RetrievalCLIPEval.py --adv 4 --gpu 0 --image_encoder ViT-B/16  ...
```

### Visual Entailment
Download SNLI-VE datasets from origin website.
```
# Attack Unimodal Embedding
python VEEval.py --adv 4 --gpu 0 --cls \
--config configs/VE.yaml \
--output_dir output/VE \
--checkpoint [Finetuned checkpoint]

# Attack Multimodal Embedding
python VEFusionEval.py ...
```

### Visual Grounding
Download MSCOCO dataset from the original website.
```
# Attack Unimodal Embedding
python GroundingEval.py --adv 4 --gpu 0 --cls \
--config configs/Grounding.yaml \
--output_dir output/Grounding \
--checkpoint [Finetuned checkpoint]

# Attack Multimodal Embedding
python GroundingFusionEval.py ...
```

## Visualization
```
python visualization.py --adv 4 --gpu 0
```
## Citation
If you find this code to be useful for your research, please consider citing.
```
@inproceedings{zhang2022towards,
  title={Towards Adversarial Attack on Vision-Language Pre-training Models},
  author={Zhang, Jiaming and Yi, Qi and Sang, Jitao},
  booktitle="Proceedings of the 30th ACM International Conference on Multimedia",
  year={2022}
}
```

## Reference
- [ALBEF](https://github.com/salesforce/ALBEF)
