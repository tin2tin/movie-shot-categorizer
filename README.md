# Shot Categorizer 🎬

<div align="center">
  <img src="https://huggingface.co/diffusers/shot-categorizer-v0/resolve/main/assets/header.jpg" width=600/>
</div><br>

Contains fine-tuning and inference code for a shot categorizer model. We fine-tuned the model on
the [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) and the
[diffusers/ShotDEAD-v0](https://huggingface.co/datasets/diffusers/ShotDEAD-v0/) dataset.

* Fine-tuned model checkpoint: [diffusers/shot-categorizer-v0](https://huggingface.co/diffusers/shot-categorizer-v0)
* Dataset: [diffusers/ShotDEAD-v0](https://huggingface.co/datasets/diffusers/ShotDEAD-v0/)
* HF Spaces for a quick try out: [ariG23498/shot-categorizer-demo](https://huggingface.co/spaces/ariG23498/shot-categorizer-demo) (thanks to @ariG23498!)

The model can be [used to curate video datasets](https://github.com/huggingface/video-dataset-scripts/tree/main/video_processing#add-shot-categories) amongst other use cases. 

## Getting started

Install PyTorch and make sure you're on a CUDA-enabled GPU. The scripts were tested on 8xH100s. 

Then install the dependencies `pip install requirements.txt`.

You can then launch training:

```bash
accelerate launch \
    --config_file=configs/accelerate-configs/ds.yaml train.py \
    --config=configs/training-configs/config.yaml
```

We provide a SLURM configuration in the `run.slurm` file.

## Inference

```bash
python inference.py --image_path=<PATH_TO_AN_IMAGE>
```

Additionally, you can run the `accuracy.py` script to compute aggregate metrics on the individual shot categories on your dataset.

## Python Lib Installation

You can install the library directly from the GitHub repository:


```bash
pip install git+https://github.com/tin2tin/movie-shot-categorizer.git
```

## Use the API 

Now you can use the library in your Python code:

```python
from movie_shot_categorizer import ShotCategorizer

# Initialize the categorizer
categorizer = ShotCategorizer()

# Categorize an image
image_path = "path/to/your/image.jpg" # Replace with the actual image path
categories = categorizer.categorize(image_path)

# Print the results
print(categories)
```

## Acknowledgement

Thanks to [this blog post](https://huggingface.co/blog/finetune-florence2) on fine-tuning Florence-2.
