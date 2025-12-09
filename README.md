# Interventional - XAI
Code related to the paper ["Locally Explaining Prediction Behavior via Gradual Interventions and Measuring Property Gradients"](https://arxiv.org/abs/2503.05424). We provide here some starting points and exemplary usage of our local interventional explanation method. The paper will be presented at WACV 2026.

## Setup

### Environment Installation
Create the conda environment using the provided `env.yaml` file:

```bash
conda env create -f env.yaml
conda activate propgrad
```

This will install all necessary dependencies including PyTorch, diffusers, gradio, and other required packages.

## Getting Started

### Tutorial Notebook
We focus here on the toy example of our paper to make it easy to use and abstract to other tasks and datasets.
We provide a tutorial Jupyter notebook that demonstrates how to:
- Load pre-trained ConvMixer classifiers (with different biases)
- Generate interventional data using InstructPix2Pix
- Measure the impact of semantic properties on model predictions
- Visualize prediction behavior shifts using our expected property gradient metric

To run the tutorial:
```bash
jupyter notebook measuring_property_impact.ipynb
```

The notebook walks through a complete example of explaining local decisions of classifiers trained on cats vs. dogs with different fur color biases.

### Gradio Demo
For an interactive demonstration, we provide a Gradio web interface:

```bash
python gradio_demo.py
```

This launches a web application where you can:
1. Upload images or use provided examples
2. Generate interventional data with custom instructions
3. Explain predictions from three toy ConvMixer models (unbiased, dark-cats-bias, dark-dogs-bias)
4. Visualize the expected property gradient and statistical significance

## Utility Functions (`propgrad/`)

The `propgrad` package provides several utility modules:

- **`convmixer.py`**: ConvMixer architecture implementation for our toy models
- **`pix2pix_interventions.py`**: Wrapper for generating interventional data using InstructPix2Pix 
- **`zollstock.py`**: Core implementation of our expected property gradient metric and shuffle test for statistical significance
- **`utils.py`**: Helper functions including:
  - `subsample_plot()`: Visualize interventional sequences
  - `ImageList_DS()`: Dataset wrapper for image lists
  - `perform_inference()`: Batch inference utilities

### Example Usage

```python
from propgrad.zollstock import Zollstock
from propgrad.pix2pix_interventions import pix2pix_edit
from propgrad.utils import ImageList_DS, perform_inference

# Generate interventional data
interventions = pix2pix_edit(
    img=example_image,
    instruction="Change the dog's fur color to white.",
    cfg_txt=np.linspace(1.01, 15.0, 100, endpoint=True),
    SEED=1337
)

# Measure property impact
z = Zollstock()
exp_prop_grad, p_value = z.shuffle_test(model_predictions)
```

## ToDos:
- currently we provide an easy to use interface for [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix) given that it is fully contained in diffusers. In our paper we mainly use [MGIE](https://github.com/apple/ml-mgie). -> Provide implementation here (more difficult setup following their original implementation).
    - this also holds for other more modern models, e.g., [FlowEdit](https://matankleiner.github.io/flowedit/), [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), [Ledits++](https://huggingface.co/spaces/leditsplusplus/project), or [FLUX.1-Kontext](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Kontext-Dev).
- Higher dimensional gradients to solve explaining for a single class
- Demo notebook and gradio setting for ImageNet classifiers, to enable explorative analysis
- efficient resampling of interventions



## If you enjoy or build on our work, consider citing our paper:

```
@inproceedings{penzel2025locally,
    author = {Niklas Penzel and Joachim Denzler},
    title = {Locally Explaining Prediction Behavior via Gradual Interventions and Measuring Property Gradients},
    year = {2025},
    doi = {10.48550/arXiv.2503.05424},
    arxiv = {https://arxiv.org/abs/2503.05424},
    note = {accepted at WACV 2026},
}
```


#### If you have any questions or want to discuss interventional XAI consider reaching out per mail: ```niklas.penzel@uni-jena.de```