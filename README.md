# Attentionproject
Rahul kuduru 700763240
## **📌 Project Overview**

This project implements the **Scaled Dot-Product Attention** mechanism using **NumPy**, following the formula used in Transformer models.

The purpose of this script is to:

* Compute attention scores
* Apply scaling
* Normalize using softmax
* Generate attention weights
* Produce the final context vector

This is a core building block of **Multi-Head Attention** and Transformer models.

## **📁 Files Included**

* **attention.py**
  Contains the full implementation of the scaled dot-product attention function along with an example test run.

## **🧠 Formula Used**

The attention is calculated using:

[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
]

Where:

* **Q** = Query
* **K** = Key
* **V** = Value
* **d_k** = Dimension of the key vector

---

## **🔧 Requirements**

Make sure Python and NumPy are installed:

```bash
pip install numpy
```

---

## **▶️ How to Run**

1. Open the project folder in VS Code
2. Open a terminal inside VS Code
3. Run the script:

```bash
python3 attention.py
```

---

## **💡 Sample Output**

Example output from the provided test:

```
Attention Weights:
 [[0.76036844 0.23963156]]

Context Vector:
 [[1.71889467 2.71889467 3.71889467]]
```

This shows:

* The model attends ~76% to the first key-value pair
* ~24% to the second
* The resulting context vector is the weighted sum of V

---

## **📜 Code Summary**

The script:

1. Computes the dot product **QKᵀ**
2. Scales by **√dₖ**
3. Applies **softmax**
4. Computes **Context = AttentionWeights × V**

The main function:

```python
def scaled_dot_product_attention(Q, K, V):
    scores = np.dot(Q, K.T)
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context_vector = np.dot(attention_weights, V)
    return attention_weights, context_vector
```
