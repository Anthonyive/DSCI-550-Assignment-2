# Large Scale Active Social Engineering Defense (ASED): Multimedia and Social Engineering

![GitHub watchers](https://img.shields.io/github/watchers/Anthonyive/DSCI-550-Assignment-2?style=social) ![GitHub Repo stars](https://img.shields.io/github/stars/Anthonyive/DSCI-550-Assignment-2?style=social) ![GitHub forks](https://img.shields.io/github/forks/Anthonyive/DSCI-550-Assignment-2?style=social)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=flat-square&logo=Jupyter)](https://jupyter.org/try) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=flat-square)](https://www.python.org/) ![commit activity](https://img.shields.io/github/commit-activity/m/Anthonyive/DSCI-550-Assignment-2?style=flat-square) ![repo size](https://img.shields.io/github/repo-size/Anthonyive/DSCI-550-Assignment-2?style=flat-square)

^ Badges will show up when the repo is publicly published.

[Quick Access to Instructions](docs/DSCI550_HW_EXTRACT_PHISHING.pdf)

## Prerequisite

1. Python virtual environment has been set up using `pipenv`. You need `pipenv` installed ([learn more about installation and usage](https://pipenv-fork.readthedocs.io/en/latest/)).
2. There are several other packages/tools you may want to use along the way. You should check out the [instruction about this assignment](docs/DSCI550_HW_EXTRACT_PHISHING.pdf)

## Usage

0. First of foremost, build up the `pipenv` environment by running `pipenv install` command in this working directory. We are using Jupyter notebooks for all of our coding, so you may want to install the ipykernel as well. To do so:

```
$ pipenv shell # this take you to the virtual environment
$ python -m ipykernel install --user --name=<my-virtualenv-name> # change the kernel name as you see fit
$ jupyter lab # run a jupyterlab instance on your localhost
```

​		Be careful about tensorflow versions. See more in [Caveats](https://github.com/Anthonyive/DSCI-550-Assignment-2#caveats) section.

1. **[Task 3]** Jupyter notebook called [3.ipynb](notebooks/3.ipynb)

   *Summary:* The notebook implements the Text-to-tag Ratio (TTR) algorithm in Tika Python that takes the XHTML representation and extracts out the relevant text. Finally, it adds a new column representing the TTR’ed resulting text.

2. **[Task 4]** Jupyter notebook called [4.ipynb](notebooks/4.ipynb)

   *Summary:* The notebook uses [GPT-2 Simple Repo](https://github.com/minimaxir/gpt-2-simple) and runs GPT-2 on TTR’d texts to generate the initial 800 emails. It trains different GPT-2 mdoels on different attack email types, so that it can emit 200 per type (Credential phishing, malware, social engineering, reconnaissance). It should generate four directories with respective types in `data/additional-features-v2/new/4_GPT-2_Generated_Text/`. Each directory should contain 200 GPT-2 generated txt files.

3. **[Task 6]** Jupyter notebooks called [6.ipynb](notebooks/6.ipynb) and [6_aggregated.ipynb](notebooks/6_aggregated.ipynb)

   *Summary:* [6.ipynb](notebooks/6.ipynb) leverages Phish Iris images dataset and DCGAN notebook from the towardsdatascience post by training a different model for each of the Phish Iris attacks (banking, social
   media, etc.). [6_aggregated.ipynb](notebooks/6_aggregated.ipynb) is a simpler version of the previous notebook. It basically creates and uses a wrapper function in `src/phishIRIS.py` for simpler applications.

4. **[Task 7]** Jupyter notebook called [7.ipynb](notebooks/7.ipynb)

   *Summary:* The notebook generates a new face for each of the 800 attack emails by applying the DCGAN technique from step 6

5. **[Task 8]** Jupyter notebook called [8.ipynb](notebooks/8.ipynb)

   *Summary:* The notebook uses [rowanz/grover repo](https://github.com/rowanz/grover) and generates new Grover model based on the extracted text from the fraudulent email corpus. Then, it uses Grover to test for falsification and to retroactively add that feature as a column to [TSV v 2 data](data/additional-features-v2/new/assignment2.tsv).

6. **[Task 9]** Jupyter notebook called [9.ipynb](notebooks/9.ipynb)

   *Summary:* The notebook generates captions for new Phish Iris images.

7. **[Task 10]** Jupyter notebook called [10.ipynb](notebooks/10.ipynb)

   *Summary:* The notebook uses GPT-2 model from huggingface to create 3 fake GPT-2 attacker-victim replies for each email.

8. **[Task 11]** Jupyter notebook called [11.ipynb](notebooks/11.ipynb)

   *Summary:* The notebook generates the new rows in [TSV v 2 data](data/additional-features-v2/new/assignment2.tsv) with your new attacks and their associated features from all prior steps.

## Caveats

1. Task 4 and Task 5_6_7 have different tensorflow requirements (1.15 and 2.2.0 respectively). Please install the version the notebook specifies.
2. Task 8 was done in Google Colab, and tensorflow requreiment is 1.14. 

## FAQ & Pull Requests

Please feel free to fork the repo and give it a pull request. If you encounter any problem, feel free to [email me](mailto:yzhang71@usc.edu).

## About

This is the assignment 2 from DSCI 550 Spring 2021 at USC Viterbi School of Engineering. This repo is collaborated by a group of six.

Team members: Zixi Jiang, Peizhen Li, Xiaoyu Wang, Xiuwen Zhang, Yuchen Zhang, Nat Zheng
