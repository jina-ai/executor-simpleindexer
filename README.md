# SimpleIndexer

`SimpleIndexer` is a memmap-based Indexer. It stores the documents in a folder that is
specified by `workspace`, which can be defined using the `uses_metas` argument as such:

```python
Flow.add(uses=SimpleIndexer, uses_metas={'workspace': 'workspace'})
```

To search documents, the `SimpleIndexer` leverages `DocumentArray`'s `match` function. 
The arguments to the `match` function are specified by `match_args`, which can be defined
using the `uses_with` argument as such:

```python
Flow.add(uses=SimpleIndexer,
         uses_with={'match_args': {'metric': 'cosine', 'use_scipy': False}})
```

For more information on the `match` function, please refer to the [documentation](https://docs.jina.ai/api/jina.types.arrays.neural_ops/).
