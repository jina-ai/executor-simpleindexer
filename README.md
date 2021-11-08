# SimpleIndexer

`SimpleIndexer` uses `DocumentArrayMemmap` for indexing `Document`. It is recommended to be used in most of the simple use cases when you have less than one million `Document`. 

`SimpleIndexer` leverages `DocumentArrayMmap`'s [`match`](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match) function and searches the `k` nearest neighbors for the query `Document` based on their `embedding` field by the brutal-force search. By default, it calculates the `cosine` distance and returns all the indexed `Document`.


## Advanced Usages

### Configure the index directory

`SimpleIndexer` stores the `Document` at the directory, which is specified by `workspace` field under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute. 

You can find how to override `metas` attributes at [docs.jina.ai](https://docs.jina.ai/fundamentals/flow/add-exec-to-flow/#override-metas-configuration)

```python
f = Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_metas={'workspace': '/my/tmp_folder'})
```


### Choose embeddings
The [recursive structures](https://docs.jina.ai/fundamentals/document/document-api/#recursive-nested-document) of Documents can be quite useful to represent the Documents at different semantic granularity. 
For example, storing a PDF file stored as a Document, 
you might have the whole PDF file stored at `granularity=0` as a `root` Document and have each sentence stored at `granularity=1` as `chunks`. In this case, the embedding is usually calculated for the sentences and therefore
you need to set `traversal_rdarray=('c',)` to choose the embeddings from `chunks` for the indexed Documents. When querying, you might want to encode the `embedding` of the query Document directly. Thereafter, you need to choose the embeddings of the from the `root` Document with `granularity=0` and set `traversal_ldrray=('r', )`. 

Both configurations can be done by overriding the `with` arguments. Find more information about the `match_args` at [here](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match)

```python
f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_with={
        'match_args': {
            'traversal_rdarray': ('c',),
            'traversal_ldarray': ('r',)}})
```

### Check embeddings
`SimpleIndexer` does NOT check the embeddings when indexing. When Documents without embedding are indexed, the whole index will be no longer usable. If you are not sure whether all the Documents have embeddings,
you can write a simple executor and uses before `SimpleIndexer` to filter out the invalid ones. In the codes below, we filter out the Documents without embeddings.

```python
from jina import DocumentArray, Executor, requests

class EmbeddingChecker(Executor):
    @requests(on='/index')
    def filter(self, docs, **kwargs):
        filtered_docs = DocumentArray()
        for doc in docs:
            if doc.embedding is not None:
                filtered_docs.append(doc)
        return filtered_docs

f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_before=EmbeddingChecker)
```

### Limit returning results  
In some cases, you will want to limit the total number of retrieved results. `SimpleIndexer` uses the `limit` argument 
from the `match` function to set this limit. Note that when using `shards=N`, the `limit=K` is the number of retrieved results for each shard and total number of retrieved results is `N*K`. For more information about shards, please read [Jina Documentation](https://docs.jina.ai/fundamentals/flow/topology/#partition-data-by-using-shards)

```python
f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_with={
        'match_args': {
            'limit': 10}})
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

**WARNING**: `SimpleIndexer` does not filter out Documents without embeddings or with embeddings of a wrong shape. If such data is indexed, the SimpleIndexer workspace will have to be deleted and re-built. Make sure your Flow filters these out with whatever business logic required.

## Used-by

- [Crossmodal Search for ImageNet](https://github.com/jina-ai/example-crossmodal-search)
- [Video Question-Answering](https://github.com/jina-ai/example-video-qa/tree/feat-simple-tutorial)
- [Video In-Content Search](https://github.com/jina-ai/example-video-search/tree/feat-simple-tutorial)
- [Similar Audio Search](https://github.com/jina-ai/example-audio-search)


## Reference
- [Indexers on Jina Hub](https://docs.jina.ai/advanced/experimental/indexers/)