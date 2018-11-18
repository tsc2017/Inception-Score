Inception Score (compatible with Tensorflow 1.6+)
=====================================
While the canonical OpenAI implementation of "Inception Score" for the evaluation of generative models is no longer compatible with new releases of Tensorflow,
a new implementation of it is provided. Also, a bug raised in [https://github.com/openai/improved-gan/issues/29](https://github.com/openai/improved-gan/issues/29) is fixed. 

## Prequisites
- `numpy` and `tensorflow-gpu`

## Features
- Fast and memory-efficient, written in a way that is similar to the original implementation
- No prior knowledge about Tensorflow is necessary to use this code
- Makes use of [TFGAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan)
- Downloads InceptionV1 automatically
- Compatibility with both Python2 and Python3

## Usage
- Call `get_inception_score(images, splits=10)`, where `images` is a numpy array with values ranging from 0 to 255 and shape in the form `[N, 3, HEIGHT, WIDTH]` where `N`, `HEIGHT` and `WIDTH` can be arbitrary. `dtype` of the images is recommended to be `np.uint8` to save CPU memory.
- A smaller `BATCH_SIZE` reduces GPU memory usage, but at the cost of a slight slowdown.
- If you want to compute a general Classifier Score with probabilities `preds` from another classifier, run `preds2score(preds, splits=10)`, where `preds` is a numpy array of arbitrary shape `[N, num_classes]`.
## Links
- The Inception Score was proposed in the paper [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- Code for the [Fréchet Inception Distance](https://github.com/tsc2017/Frechet-Inception-Distance)
