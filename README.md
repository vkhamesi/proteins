# Protein Function Prediction using Deep Learning

## Context
Protein function prediction is a critical problem in bioinformatics. It involves determining the specific biological functions of a given protein based on its sequence (amino acids) and structure. This is known to be a challenging task due to the complex relationships between amino acids and their functional roles, as well as the vast number of possible protein functions. Accurate predictions hold immense potential for applications in drug discovery, understanding disease mechanisms, how organs and tissues work, and more.

DeepMind has made a great introduction video on why proteins are useful in any biological process.

[![](http://img.youtube.com/vi/KpedmJdrTpY/mqdefault.jpg)](http://www.youtube.com/watch?feature=player_embedded&v=KpedmJdrTpY)

## Methdology

### Proposed Architecture
The proposed model takes two inputs, a protein amino acids sequence and a protein function description, and is optimised to learn patterns and interactions between the two inputs to estimate the probability of the given protein to have the given function capability. Many so-called Large Protein Models have been proposed in the literature to extract features from proteins amino acids sequences, often trained on an unmasking tokens (amino acids) task. Similarly, many domain experts Large Language Models have been explored in the literature, including biology, in order to understand and extract feature from a technical biology text input, also pre-trained on an unmasking task and sometimes fine-tuned on a next sentence prediction task. In this work, we chose to use ProtBERT (published by RostLab) as a Large Protein Model and BioBERT (published by DMIS Lab) as a Large Language Model. 

The proposed model is similar to a Sentence-BERT architecture: the model computes embeddings for a given protein amino acids sequence using ProtBERT and for a protein function description using BioBERT. We then reshape both embeddings so that they have exact same size. The embedding fusion strategy is similar to Sentence-BERT: we concatenate the embeddings as well as their absolute difference, and we use this new concatenated object as the input of a fully connected Multi Layer Perceptron with single output, the probability (or logit) of match between the given protein and the given function. The heuristic behind the model architecture is simple: ProtBERT extracts information about the protein sequence, BioBERT extracts information about the function description, and we also add this absolute difference term to increase the overall capability of the model to estimate the interaction between the given protein and the given function. 

The model architecture is summarised below.

<p align="center">
    <img src="https://github.com/vkhamesi/proteins/blob/main/img/architecture1.svg" alt>
</p>

### Making Fine-Tuning efficient
To effectively train a model with approximately 1 billion parameters on a single T4 GPU in Google Colab, we employed the following key techniques.

**1. Mixed Precision & Quantization.**
Quantization is a technique that involves representing the model's weights and activations with lower precision numbers. This reduces the memory footprint and computational requirements, allowing for faster training and inference. Mixed precision training combines both lower and higher precision data types to strike a balance between precision and efficiency. 

**2. Model Distillation.**
Distillation is a knowledge transfer technique where a larger, more complex model (teacher) imparts its knowledge to a smaller, more efficient model (student). This is achieved by training the student model to mimic the output probabilities of the teacher model. Distilling a model from scratch is a long task on its own, so we decided to load pre-existing distilled models of ProtBERT and BioBERT available on HuggingFace.

**3. Fine-Tuning using Low Rank Adaptation.**
LoRA (Low-Rank Adaptation of Large Language Models) is a widely used parameter-efficient fine-tuning technique. It consists in freezing the weights of the layer, and injects trainable rank-decomposition matrices. Assume we have an $n \times n$ pre-trained dense layer (or weight matrix), $W_0$. We initialize two dense layers, $A$ and $B$, of shapes $n \times r$, and $r \times n$, respectively, where the rank $r$ is much smaller than $n$. The equation is $y = W_0 x + b_0 + B A x$. Therefore, LoRA assumes that the original dense layer from the pre-trained model has intrinsic low rank because LLMs are over-parametrised, so it can be factorised in two low-rank matrices. Overall, this reduces by a large factor the number of parameters to be trained or fine-tuned, and it also reduces the required amount of memory.

## Technical details

- We use an open-source distilled version of [ProtBERT](https://huggingface.co/Rostlab/prot_bert) model available on HuggingFace as [DistilProtBERT](https://huggingface.co/yarongef/DistilProtBert). We use an open-source distilled version of [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) model available on HuggingFace as [DistilBioBERT](https://huggingface.co/nlpie/distil-biobert). 
- The original dataset contains $n \approx 140,000$ protein sequences and $p \approx 40,000$ function descriptions. However, we trained our models on only a few thousand pairs using different sampling strategies.
- The maximum sequence length for the protein model is set to $1024$ and the maximum sequence length for the large language model is set to $256$.
- We trained models on a single T4 GPU using Google Colab using a batch size of 32 and mixed precision (`float16`).
- We fine-tune both pre-trained models using PEFT and LoRA: 3 layers (out of 6) of BioBERT model are fine-tuned, and 2 (out of 15) layers of ProtBERT are fine-tuned. The reshaping layers and the final multi-layer perceptron are trained from scratch using ReLU activations and dropout as a regularisation technique. In total, approximately 0.50% of the model parameters are trainable. 
- The optimiser is AdamW with a learning rate of $10^{-3}$. The loss function is a weighted binary cross-entropy from logits to manage class imbalance (in average, a protein has less than 10% of all functions). Each model is trained on 3 epochs.

## Results

$$
\begin{aligned}
& \begin{array}{cccccccc}
& n_{\text{prot.}} & n_{\text{func.}} & \text{Seen prot.} & \text{Seen func.} & \text{AUC} & \text{Precision} & \text{Recall} & F_1\text{-Score} \\
\hline
\text{Imbalanced} & 500 & 20 & ✔ & ✔ & 0.75 & 0.16 & 0.47 & 0.24 \\
 &  &  & ✔ & ✗ & 0.52 & 0.08 & 0.04 & 0.06 \\
 &  &  & ✗ & ✔ & 0.72 & 0.17 & 0.47 & 0.24 \\
 &  &  & ✗ & ✗ & 0.49 & 0.07 & 0.03 & 0.04 \\
\hline
\text{Imbalanced} & 20 & 500 & ✔ & ✔ & 0.88 & 0.32 & 0.79 & 0.45 \\
 &  &  & ✔ & ✗ & 0.77 & 0.25 & 0.58 & 0.35 \\
 &  &  & ✗ & ✔ & 0.74 & 0.24 & 0.58 & 0.34 \\
 &  &  & ✗ & ✗ & 0.69 & 0.22 & 0.46 & 0.29 \\
\hline
\text{Balanced} & 100 & 100 & ✔ & ✔ & 0.83 & 0.26 & 0.71 & 0.38 \\
 &  &  & ✔ & ✗ & 0.62 & 0.13 & 0.41 & 0.19 \\
 &  &  & ✗ & ✔ & 0.79 & 0.26 & 0.65 & 0.37 \\
 &  &  & ✗ & ✗ & 0.61 & 0.12 & 0.40 & 0.19 \\
\hline
\text{Balanced} & 200 & 200 & ✔ & ✔ & 0.84 & 0.26 & 0.71 & 0.38 \\
 &  &  & ✔ & ✗ & 0.72 & 0.25 & 0.68 & 0.37 \\
 &  &  & ✗ & ✔ & 0.82 & 0.27 & 0.69 & 0.39 \\
 &  &  & ✗ & ✗ & 0.70 & 0.26 & 0.65 & 0.37 \\
\hline
\end{array}
\end{aligned}
$$

The model reaching best performance on test data is the proposed architecture trained on 200 protein sequences and 200 function descriptions, that is 4 times more training pairs than any other trained model. Training this model takes approximately 3 hours on a single T4 GPU, whereas other models took around 40 minutes to be trained. Note however that this model is closely followed by the imbalanced model trained on 20 protein sequences and 500 function descriptions, suggesting that the original distilled ProtBERT model seems powerful enough to extract useful information from protein sequences, while the original distilled BioBERT model requires more fine-tuning. This could also be explained by the fact that function embeddings are only extracted from their text description and which could be lacking of accurate information. 

All models are available to be downloaded from the following links:
- model trained on 500 proteins and 20 functions `model_500_20.pt` [here]()
- model trained on 20 proteins and 500 functions `model_20_500.pt` [here]()
- model trained on 100 proteins and 100 functions `model_100_100.pt` [here]()
- model trained on 200 proteins and 200 functions `model_200_200.pt` [here]()

You can load the models directly as PyTorch models as
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("path/to/model_n_p.pt", map_location=device)
model.eval()
```

## Conclusion

Our approach represents a novel end-to-end architecture inspired by recommender systems (two-towers) which demonstrates the ability to learn and generalize to both previously unseen protein sequences and novel functions, showcasing its significant adaptability, while simply fine-tuned on a single T4 GPU on Google Colab. Protein function prediction is crucial for understanding the biological roles of proteins, which are fundamental building blocks of life. It enables researchers to uncover the specific tasks and interactions that proteins perform within cells, shedding light on disease mechanisms and aiding in the development of targeted therapies. Additionally, accurate predictions have significant implications for drug discovery, as they guide the identification of potential drug targets and the design of effective pharmaceutical interventions.

In terms of further research, it would obviously be interesting to observe the behaviour of the proposed approach at a larger scale, such as a better GPU (e.g. A100), more training examples, or training for more epochs. More importantly, we did not leverage the graph structure of protein functions. Indeed, some protein functions are related to each other in the sense that they would be close, and therefore if a protein has a specific function, it may be likely that it has other similar functions. We could therefore have included in our approach the graph structure of protein functions and instead of only using the function description, we would have used its node representation in the global graph. The proposed model architecture is shown below.

<p align="center">
    <img src="https://github.com/vkhamesi/proteins/blob/main/img/architecture2.svg" alt>
</p>

However, note that the explored approach is more flexible than this proposal in the sense that using the graph structure of protein functions would not allow possibly new functions that are not represented in the graph. Indeed, the current architecture only requires the protein amino acids sequence and a text description of the function to be evaluated, so it could be extended to any future and possibly unknown (yet) proteins and functions. Therefore, specifying a protein sequence and a function description is similar to prompting the model. 
