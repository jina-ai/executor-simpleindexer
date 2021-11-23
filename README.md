# SimpleIndexer

`SimpleIndexer` uses `DocumentArrayMemmap` for indexing `Document`. It is recommended to be used in most of the simple use cases when you have less than one million `Document`. 

`SimpleIndexer` leverages `DocumentArrayMmap`'s [`match`](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match) function and searches the `k` nearest neighbors for the query `Document` based on their `embedding` field with a naive / brute force approach. By default, it calculates the `cosine` distance and returns all the indexed `Document`.


## Advanced Usages

### Configure the index directory

`SimpleIndexer` stores the `Document` at the directory, which is specified by `workspace` field under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute. 
You can override the default configuration as below,

```python
f = Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_metas={'workspace': '/my/tmp_folder'})
```

Find more information about how to override `metas` attributes at [Jina Docs](https://docs.jina.ai/fundamentals/flow/add-exec-to-flow/#override-metas-configuration)


### Choose embeddings
The [recursive structures](https://docs.jina.ai/fundamentals/document/document-api/#recursive-nested-document) of Documents can be quite useful to represent the Documents at different semantic granularity. 
For example, indexing PDF files stored as Documents, 
you might have the whole PDF file stored as a `root` Document and have each sentence stored as `chunks`. Assuming that the embeddings are calculated for the sentences, 
you will want to choose the embeddings from `chunks` of the indexed Documents when comparing with the query embeddings. This can be configured by setting `traversal_rdarray='c'`. `'c'` denotes the `chunks`. 
As for the query Documents, you will usually to use the `embedding` of the `root` Document and set `traversal_ldrray='r'`. `'r'` denotes the `root`. 

By default, both `traversal_ldarray` and `traversal_rdarray` is set to `'r'` so that the embedding is retrieved from `root` from both the querying and indexed Document. Both configurations can be done by overriding the `with` arguments. Find more information about the `match_args` at [here](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match).

```python
f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_with={
        'match_args': {
            'traversal_rdarray': 'c',
            'traversal_ldarray': 'r'}})
```

### Check embeddings

> **WARNING**: `SimpleIndexer` does not filter out Documents without embeddings or with embeddings of a wrong shape. If such data is indexed, the SimpleIndexer workspace will have to be deleted and re-built. Make sure your Flow filters these out with whatever business logic required.

If you are not sure whether all the Documents have valid embeddings,
you can write a simple executor and uses before `SimpleIndexer` to filter out the invalid ones. In the codes below, we filter out the Documents without embeddings.

```python
from jina import DocumentArray, Executor, requests

EMB_DIM = 512

class EmbeddingChecker(Executor):
    @requests(on='/index')
    def check(self, docs, **kwargs):
        filtered_docs = DocumentArray()
        for doc in docs:
            if doc.embedding is None:
                continue
            if doc.embedding.shape[0] != EMB_DIM:
                continue
            filtered_docs.append(doc)
        return filtered_docs

f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_before=EmbeddingChecker)
```

### Limit returning results  
In some cases, you will want to limit the total number of retrieved results. `SimpleIndexer` uses the `limit` argument 
from the `match` function to set this limit. Note that when using `shards=N`, the `limit=K` is the number of retrieved results for **each shard** and total number of retrieved results is `N*K`. By default, `limits` is set to `20`. For more information about shards, please read [Jina Documentation](https://docs.jina.ai/fundamentals/flow/topology/#partition-data-by-using-shards)

```python
f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_with={'match_args': {'limit': 10}})
```


### Configure the other search behaviors

You can use `match_args` argument to pass arguments to the `match` function as below.

```python
f =  Flow().add(
     uses='jinahub://SimpleIndexer',
     uses_with={
         'match_args': {
             'metric': 'euclidean',
             'use_scipy': True}})
```

- For more details about overriding [`with`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#yaml-interface) configurations, please refer to [here](https://docs.jina.ai/fundamentals/flow/add-exec-to-flow/#override-with-configuration).
- You can find more about the `match` function at [here](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match)



### Configure the Search Behaviors on-the-fly

**At search time**, you can also pass arguments to config the `match` function. This can be useful when users want to query with different arguments for different data requests. For instance, the following codes query with a custom `limit` in `parameters` and only retrieve the top 100 nearest neighbors. 


```python
with f:
    f.search(
        inputs=Document(text='hello'), 
        parameters={'limit': 100})
```


## Used-by

- [Crossmodal Search for ImageNet](https://github.com/jina-ai/example-crossmodal-search)
- [Video Question-Answering](https://github.com/jina-ai/example-video-qa/tree/feat-simple-tutorial)
- [Video In-Content Search](https://github.com/jina-ai/example-video-search/tree/feat-simple-tutorial)
- [Similar Audio Search](https://github.com/jina-ai/example-audio-search)


## Reference
- [Indexers on Jina Hub](https://docs.jina.ai/advanced/experimental/indexers/)