# SimpleIndexer

`SimpleIndexer` is a [mmap](https://docs.python.org/3/library/mmap.html)-based Indexer. 
`SimpleIndexer` is recommended to be used when you have less than one million Documents. 


It stores the [Document](https://docs.jina.ai/fundamentals/document/document-api/)s at the directory specified by `workspace`, 
which can be defined using the `uses_metas` argument as such:

```python
Flow().add(
    uses=SimpleIndexer,
    uses_metas={'workspace': './foo_workspace'})
```

To search documents, the `SimpleIndexer` leverages [`DocumentArrayMmap`](https://docs.jina.ai/fundamentals/document/documentarraymemmap-api/)'s `match` function. 
For advanced usages, one can pass arguments to the `match` function by specifying `match_args` as below:

```python
Flow().add(
    uses=SimpleIndexer,
    uses_with={
        'match_args': {
            'metric': 'cosine',
            'use_scipy': False}})
```

## Used-by
- [Semantic Wikipedia Search with Transformers](https://github.com/jina-ai/examples/tree/master/wikipedia-sentences)
- [Find Similar Audio Clips](https://github.com/jina-ai/examples/tree/master/audio-to-audio-search)

## Reference
- For more information on the `match` function, please refer to the [documentation](https://docs.jina.ai/api/jina.types.arrays.neural_ops/#jina.types.arrays.neural_ops.DocumentArrayNeuralOpsMixin.match).
