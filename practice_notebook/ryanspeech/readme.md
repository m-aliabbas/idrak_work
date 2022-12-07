Experiment: IdrakAI 

#Dataset Operation

1. Login to Hugging face `huggingface-cli login`
2. Download the dataset. Preprocess it i.e. Unpacking, Cleaning , Resampling.
3. Making Your Dataset Repo to Hugging Face
4. Packing Your Data 
5. Upload Data

For all these operations use the notebook 

```exp_ryanspeech_dataset_operations.ipynb```

#Training The ASR

We are using Pytorch Wrapper of Huggingface. For Finetuning Wave2Vec 2.0 Base on Dataset procsssed above please refer
``` exp_wav2vec_ryanspeech.ipynb ``` 

To make a subsample of data use [:16] for selecting 16 samples. Edit these lines i.e line 9

```
common_voice_train = load_dataset(dataset_repo, split="train[:16]")
common_voice_test = load_dataset(dataset_repo, split="test[:4]")

```

Also, change `repo_name` to your hugging face repo name i.e line 28
```
repo_name = "idrak_wav2vec_ryan_voice_subset"
```

