# SimpleIndexer

`SimpleIndexer` uses an `SQLite`  database for indexing `Document`. It is recommended to be used in most of the simple use cases when you have less than one million `Document`. 

`SimpleIndexer` leverages `DocumentArray`'s [`match`](https://docs.jina.ai/api/jina.types.arrays.mixins.match/?module-jina.types.arrays.mixins.match) function and searches the `k` nearest neighbors for the query `Document` based on their `embedding` field with a naive / brute force approach. By default, it calculates the `cosine` distance and returns all the indexed `Document`.


## Advanced Usages

### Configure the index directory

The sqlite database file in which the `SimpleIndexer` stores the `Document` could be specified by  the`workspace` field under the `metas` attribute. 
The table in which it is stored could also be specified by the `table_name` filed under the `uses_with`parameters. By default the table name is random.
You can override the default configuration as below,

```python
f = Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_metas={'workspace': '/my/tmp_folder'},
    uses_with = {'table_name': 'my_custon_table_name'}
    )
```

Find more information about how to override `metas` attributes at [Jina Docs](https://docs.jina.ai/fundamentals/flow/add-exec-to-flow/#override-metas-configuration)

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

- For more details about overriding configurations, please refer to [here](https://docs.jina.ai/fundamentals/executor/executor-in-flow/#special-executor-attributes).
- You can find more about the `match` function at [here](https://docarray.jina.ai/api/docarray.array.mixins.match/#docarray.array.mixins.match.MatchMixin.match)

### Traversing Documents

In order to traverse at different levels when you `match`, DocumentArray allows the `@` syntax. 

For the SimpleIndexer, this is exposed via the parameters:

- `traversal_right`. How to traverse the DocumentArray **inside** the Indexer
- `traversal_left`. How to traverse the DocumentArray you **search with**

The Indexer has both of these set to the default of `@r`. These can be overridden at Executor initialization time or at search time.

Example initialization:

```python
f =  Flow().add(
    uses='jinahub://SimpleIndexer',
    uses_with={
        'traversal_right': '@c',
        'traversal_left': '@r'
    })
```

Example search:

```python
from jina import Client

Client().search(
    inputs=[Document(text='hello')], 
    parameters={
        {'traversal_left': '@r',
        'traversal_right': '@r'}
    })
```

The above will search on root level for both your search docs and the indexed docs.

If you want to match on the 1st chunk level on the docs inside the Indexer, use 

```python
    parameters={
        'traversal_right': '@c',
        'traversal_left': '@r'
        }
```

For a full guide on the syntax, check [here](https://docarray.jina.ai/fundamentals/documentarray/matching/).

### Configure the Search Behaviors on-the-fly

**At search time**, you can also pass arguments to config the `match` function. This can be useful when users want to query with different arguments for different data requests. For instance, the following codes query with a custom `limit` in `parameters` and only retrieve the top 100 nearest neighbors. 


```python
with f:
    f.search(
        inputs=Document(text='hello'), 
        parameters={'limit': 100})
```



### Clear the indexer

You can easily clear the indexer by calling the `/clear` endpoint

```python
with f:
    f.post('/clear')
```

## Used-by

- [Crossmodal Search for ImageNet](https://github.com/jina-ai/example-crossmodal-search)
- [Video Question-Answering](https://github.com/jina-ai/example-video-qa/tree/feat-simple-tutorial)
- [Video In-Content Search](https://github.com/jina-ai/example-video-search/tree/feat-simple-tutorial)
- [Similar Audio Search](https://github.com/jina-ai/example-audio-search)

