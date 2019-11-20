# Training

## Python Environment

## Requirements


```
pip install gensim
```

## Train the model

1. The script supports training from a single text file or directory of files. Create a text file or folder of multiple files. Now run `train.py` with the name of the file or folder.

Example:

```
python train.py file.xt
python train.py files/
```


3. The script will output a `vectors.txt` and `vectors.json` file, however, if you would like to specify an output file name you can use the additional argument `-o` for that.

```
python train.py data.txt -o output.json
```

4. The output JSON file  can be used now with the [ml5.js word2vec examples](https://github.com/ml5js/ml5-examples/tree/master/p5js/Word2Vec).
