# :detective: Detecting Pretraining Data from Large Language Models

This repository provides an original implementation of [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2308.04430) by *[Weijia Shi](https://weijia-shi.netlify.app), *Anirudh Ajith, [Mengzhou Xia](https://xiamengzhou.github.io), [Yangsibo Huang](https://hazelsuko07.github.io/yangsibo), [Daogao Liu](https://daogaoliu.github.io/)
, [Terra Blevins](https://blvns.github.io/)
, [Danqi Chen](https://www.cs.princeton.edu/~danqic)
, [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz).

<p align="center">
  <img src="mink_prob.png" width="80%" height="80%">
</p>

:star: If you find our implementation and paper helpful, please consider citing our work :star: :

```bibtex
@article{ silo,
    title={ {SILO} Language Models: Isolating Legal Risk in a Nonparametric Datastore },
    author={ Min, Sewon and Gururangan, Suchin and Wallace, Eric and Hajishirzi, Hannaneh and Smith, Noah and Zettlemoyer, Luke },
    year={ 2023 },
    journal={ arXiv preprint arXiv:2308.04430 },
    url={ https://arxiv.org/abs/2308.04430 }
}
```

## 📘 WikiMIA Datasets

Access our **WikiMIA datasets** directly on [Hugging Face](https://huggingface.co/datasets/swj0419/WikiMIA).

#### Loading the Datasets:

```python
from datasets import load_dataset
LENGTH = 64
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
```
* Available Text Lengths: `32, 64, 128, 256`.
* *Label 0*: Refers to the unseen data during pretraining. *Label 1*: Refers to the seen data.
* WikiMIA is applicable to all models released between 2017 to 2023 such as  `LLaMA1/2, GPT-Neo, OPT, Pythia, text-davinci-001, text-davinci-002 ...`

## 🚀 **Run our Min-K% Prob & Other Baselines**

Our codebase supports many models: Whether you're using **OpenAI models** that offer logits or models from **Huggingface**, we've got you covered:

- **OpenAI Models**:
  - `text-davinci-003`
  - `text-davinci-002`
  - ...

- **Huggingface Models**:
  - `meta-llama/Llama-2-70b`
  - `huggyllama/llama-70b`
  - `EleutherAI/gpt-neox-20b`
  - ...

🔐 **Important**: When using OpenAI models, ensure to add your API key at `Line 38` in `run.py`:
```python
openai.api_key = "YOUR_API_KEY"
```
Use the following command to run the model:
```bash
python src/run.py --target_model text-davinci-003 --ref_model huggyllama/llama-7b --data swj0419/WikiMIA --length 64
```
🔍 Parameters Explained:
* Target Model: Set using --target_model. For instance, --target_model huggyllama/llama-70b.

* Reference Model: Defined using --ref_model. Example: --ref_model huggyllama/llama-7b.

* Data Length: Define the length for the WikiMIA benchmark with --length. Available options: 32, 54, 128, 256.

📌 Note: By default, outputs are saved to the out directory.

📊 Baselines: Our tool comes with the following baselines: PPL, Calibration Method, PPL/zlib_compression, PPL/lowercase_ppl

