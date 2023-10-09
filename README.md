# Detecting Pretraining Data from Large Language Models
This includes an original implementation of **[Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2308.04430)** by [Weijia Shi](https://weijia-shi.netlify.app)*, Anirudh Ajith*, [Mengzhou Xia](https://xiamengzhou.github.io), [Yangsibo Huang](https://hazelsuko07.github.io/yangsibo), [Daogao Liu](https://daogaoliu.github.io/), [Terra Blevins](https://blvns.github.io/), [Danqi Chen](https://www.cs.princeton.edu/~danqic), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz).

<p align="center">
  <img src="mink_prob.png" width="80%" height="80%">
</p>

If you find our code and paper helpful, pleace cite our paper. 
```
@article{ silo,
    title={ {SILO} Language Models: Isolating Legal Risk in a Nonparametric Datastore },
    author={ Min, Sewon and Gururangan, Suchin and Wallace, Eric and Hajishirzi, Hannaneh and Smith, Noah and Zettlemoyer, Luke },
    year={ 2023 },
    journal={ arXiv preprint arXiv:2308.04430 },
    url={ https://arxiv.org/abs/2308.04430 }
}
```

## Data
Pleaase find our WikiMIA datasets [here](https://huggingface.co/datasets/swj0419/WikiMIA). You can load the datasets from huggingface via
```python
from datasets import load_dataset
LENGTH=64
dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{LENGTH}")
```
Our datasets have subsets of different-length texts ranging from `LENGTH=32/64/128/256`.  

(Label 0 refers to the unseen data, whereaswhe Label 1 refers to the seen data.)

## Run our method Min-K% Prob and other baselines (PPL, Calibration method, PPL/zlib_compression, PPL/lowercase_ppl)
Our codebase supports both OpenAI models that provide logits such as `text-davinci-003`, `text-davinci-002` or models that can be loaded from Huggingface such as `meta-llama/Llama-2-70b`, `huggyllama/llama-70b`. 
When using OpenAI models, don't forget to put your API keys in Line 38 in `run.py`: `openai.api_key = "YOUR_API_KEY"`
```bash
python src/run.py --target_model text-davinci-003 --ref_model huggyllama/llama-7b --data swj0419/WikiMIA --length 64
```
* You can specify the target model for the derection by `--target_model text-davinci-003 or huggyllama/llama-70b`. You can also specify the reference model used for calibraiton by `--ref_model huggyllama/llama-7b`.
* You can specify the length of the WikiMIA benchmark you want to evaluate on using `--length 32 or 54 or 128 or 256`.
* By default, it will save the output to `out`




