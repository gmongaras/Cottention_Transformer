# Cottention_Transformer




# Log
## BERT
### Initial
Initially, both softmax and cosine did quite bad. However, cosine would not converge at all.

### Attempting to fix Cosine Attention
Below are the attempts at trying to fix the issue of cosine attention divergence
- Gradient clipping - Kind of stabalized training, but was terrible
- Higher weight decay (0.1) - Did not help at all
- Lower learning rate (1e-5) - Appeared to stabalize training

At this point, training was tested with both softmax and cosine again, with cosine winning at 1e-5 learning rate.

### AdamW
After "fixing" cosine attention, it was realized that Adam was used instead of AdamW. After using AdamW, softmax at 1e-5 and 1e-4 learning rate completely destroyed cosine attention.

### Improving Convergence of Cosine Attentoin
Seeing Cosine attention converge gave us hope that it could be optimized to match normal attention
- Slaping ReLU around the "attention" matrix - In normal ReLU fashion, this worked quite well and convergence was much faster. We hypothesize this could be because the model could "throw away" tokens when doing linear combinations with the values.
- Other activations didn't do as well, even variations of ReLU which is probably because tokens cannot be throw away.
- Slightly higher learning rate - Learning rates of 3e-5 and 5e-5 diverged, 1e-5 seemed to be the sweet spot.
- Token Dropout - Dropout in the attention matrix made the model slightly worse.
- Angle dist - Since ReLU worked, why wouldn;t the angle distance work. This is basically an unsigned variant of cosine similarity. This was much worse than cosine similarity with ReLU.

### Stabalizing Traning At a Higher Learning Rate
We realized to make cosine attention match that of softmax, the learning rate must be increased. Since ReLU was so good, why not have it do a little better
- Learning ReLU - Can ReLU do better if parts of it such as the slope or cutoff are learned? Apparently not as this makes the model worse.
- "Soft ReLU" - We noticed several of the attention maps using ReLU were dead. This is a variant on ReLU where the forward pass is normal ReLU, but backward pass is leaky ReLU, thus creating some intermediate between leaky and normal ReLU. This didn't work :(

### Measuring Magnitude
What's the one thing attention has over the current method? The magnitude of attention is at most 1. The magnitude of cosine similarity is at most the number of tokens in the sequence. This must blow up the output values if the attention map has large values.

Looking at the magnitude of the cosine scores makes it obvious that this is the problem. The magnitudes continuously increase as training goes on, reaching values of above 10.

### Fixing Magnitude Issues
Fixing this issue is quite easy, just normalize the attention matrix. To do this, we can divide by the sequence length. This means the magnitude will never be greater than 1. Doing this fixes the problem. The model can now be trained at a high learning rate and this method also beats ReLU at 1e-5 learning rate.

The best part is this method has linearity:

(N(Q)@N(K^T))/s @ V = N(Q)@N(K^T)) @ V/s = N(Q) @ (N(K^T) @ V/s)

### More Improvements
Slight improvments can be made by changing the exponent of the dividend, that is of the sequence length scaling. Using a square root seems to work better, but also leads to loss explosion at a point.

## GPT



# Tests
## Activation Functions
Try different activations functions on the attention matrix:
- ReLU
- Sigmoid
- Softmax???


## Similarity Scoring
Try different similarity measures to obtain the attention matrix:
- Cosine similarity - cosFormer/cottention
- Euclidean distance - Euclidformer
- Manhattan distance - ManFormer?


## Learnable Similarity?
What if we have the model learn the similarity function?
- Ex: learnable Chebyshev p value.
