r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["seq_len"] = 64
    hypers["h_dim"] = 256
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.2
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.5
    hypers["lr_sched_patience"] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
There are two major reasons we saw in lecture 8 slides to why it is better to split the corpus into sequences instead of 
training on the whole text.
Firstly and foremost, to train a GRU we have to use backpropagation through time i.e. forward through the entire 
sequence to compute the loss, then backward through the entire sequence to compute gradient. This action scales linearly
with the length of the sequence making it practically impossible for the text we are using.
Secondly, we face the recurring problem of the 'vanishing/exploding gradients'. As we saw in the previous HW a model with 
many layers is prone to suffer from the 'vanishing gradient' problem (or exploding but in that case we can address the 
problem using methods like gradient clipping). The GRU forms a layer structure throughout time thus, making every 
character a new layer and forms a very deep network. 
Splitting the corpus to more manageable sizes helps us limit the memory we need to backpropagate (makes it linearly 
dependant on the sequence size) and also makes the backpropagation chunk be smaller and thus less prone to vanishing 
gradients.

"""

part1_q2 = r"""
The reason why the generated text clearly shows memory longer than the sequence although we are truncating the gradients
is that every sequence uses the last sequence's output hidden state as its initial hidden state. 
This way, information propagates through the different sequences ideally forming a compressed representation of all the
seen data up to that point.
"""

part1_q3 = r"""
The reason why we are not shuffling the order of batches when training is that as stated above, GRU's rely on the 
initial hidden state to get a broader context. Shuffling the batches will cause sequences to get an initial hidden state
that poorly represents the context state of the input. It might get a hidden state representing much earlier parts
of the text or even parts of the text that will only happen in the future (spoiler alert).
By keeping the batches ordered we ensure that the start of every sequence is the exact hidden state of the previous 
sequence ensuring coherent context propagation.
"""



part1_q4 = r"""
1) We lower the temperature for sampling because we want the model to be more confident in its prediction. 
This way the model will have a higher probability of choosing the most likely character

2) When the temperature is very high ($T \to \infty$), the terms inside the exponent approach zero ($\frac{y}{T} \to 0$).
Thus, $e^{y/T} \to 1$ for all logits.This results in a uniform distribution where every character has probability 
$\frac{1}{n}$, causing the model to pick characters completely at random (gibberish).Formally:
$$\lim_{T \to \infty} \frac{e^{y_i/T}}{\sum_k e^{y_k/T}} = \frac{1}{\sum_{k=1}^{n} 1} = \frac{1}{n}$$

3) When the temperature is very low ($T \to 0$), the probability distribution effectively becomes the argmax function
and the model will allways choose the most likely character resulting in a determinstic non creative output (making it
also worthless)
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
