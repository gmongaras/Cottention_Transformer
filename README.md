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

### Fixing Magnitude Issues 2
The old method works well where the attention scores are divided by the sequence length, however this leads to a problem. If the cosine similarity scores of the sequence are < 1, then the outputs will slowely converge to 0 as the number of layers increases. Alternatively, if the scroes of the sequence are > 1, then the outputs will quickly diverge and go to infinity. The current situation is much better than going to infinity, however it is far from optimal.

How about we take an exponential of the sequence length? So instead the sequence is divided by the square root of the sequence length. This works and the model does slightly better, however this diverges at a point, likely as the model has magnitudes > 1.

How about allowing the model to learn an exponential of the sequence length. In this case, the attention scoes will be divided by the sequence length to the power of a learnable constant (one for each head). This seems to work quite well, however rarely this method also dies and goes to infinity. FOrcing the model to have a value between 0 and 1 via sigmoid helps, but still runs into divergence problems.

How about just normalizing the values. So the output becomes: O = N(Q) @ N(K^T) @ N(V). Maybe this will helps. It's kind of like just messing around the unit sphere though.

How about adding a penalty to the loss? For example we could:
1. Penalize the output of the attention mechanism for have output tokens with high magnitude from the values. Note that instead of doing this for the input, we do this with the values. The Q and K are just going to be between -1 and 1, so that magnitude doesn't matter. The magnitude of the values on the other hand should be close to the magnitude of the output so the model doesn't blow up. This means token 0 in the values should have a magnitude close to token 0 in the output
2. A slightly less extreme penalty is that the average magnitude of the values should be close to the average magnitude of the output.
- In the end, penalty wouldn't covnerge at all :(

What worked best?
- Although dividing by a constant worked most of the time, it failed sometimes. Normalizing the values worked for all tests so far.


## Data Problems
Training the above was trained on data without punctuation as the data script removed these by mistake. Adding them back results in proper data. Additionally, this data had a few problems, which were addressed to clean the data further and properly.

## Fixing Stability
With the new data, the stability becomes wacky again. The value norm method dies because the magnitude of the vectors becomes too large. The norm method worked for the other data likely because that data had much shorter sentences. However, as the sentence size grows, the magnitude of the attention row vectors increases as the max magnitude is the sequence length itself. This results in an unstable behavior, as the number of layer increases, the magnitude of the vectors increases if the attention vectors have a magnitude greater than 1, which is very likely. This magnitude issue results in unstable values and unstable and large gradients.

To fix this, we are going back to the divide by the sequence lenght method for now. This method will obviously fix the magnitude issue in all cases, however, the model may hae issues as the scores could decrease to 0 instead of increasing to infinity, which is a much better problem. We will see how this works out. 

I am afraid that this method may result in long sequence issue. Let's say we have a sequence of length S and we increase the sequence to 2S. Let's say that the addition of S tokens doesn't add any attention scores as perhaps we addeed some trash token the model doesn't care about. Then, the attention scores are divided by two, which may cause vanishing values as layers are stacked. I am thinking that since information among tokens is sparse, as the sequence length increases, the attention scores will have issues. Just a hypothesis, not sure if this is actually a problem or not.



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
