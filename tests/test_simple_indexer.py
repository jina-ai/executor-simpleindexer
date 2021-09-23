import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow

from executor import SimpleIndexer


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([1, 0, 0, 0])),
            Document(id='doc2', embedding=np.array([0, 1, 0, 0])),
            Document(id='doc3', embedding=np.array([0, 0, 1, 0])),
            Document(id='doc4', embedding=np.array([0, 0, 0, 1])),
            Document(id='doc5', embedding=np.array([1, 0, 1, 0])),
            Document(id='doc6', embedding=np.array([0, 1, 0, 1])),
        ]
    )


@pytest.fixture
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([0, 0, 0, 1])),
        ]
    )


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[1] / 'config.yml'))
    assert ex._match_args == {}


def test_flow(tmpdir):
    f = Flow().add(
        uses=SimpleIndexer,
        uses_metas={'workspace': str(tmpdir)},
    )

    with f:
        resp = f.post(
            on='/index',
            inputs=[Document(id='a', embedding=np.array([1]))],
            return_results=True,
        )
        resp = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1]))],
            return_results=True,
        )
        assert resp[0].docs[0].matches[0].id == 'a'


def test_fill_embeddings(tmpdir):
    metas = {'workspace': str(tmpdir)}
    indexer = SimpleIndexer(metas=metas)

    indexer.index(DocumentArray([Document(id='a', embedding=np.array([1]))]))
    search_docs = DocumentArray([Document(id='a')])
    indexer.fill_embedding(search_docs)
    assert search_docs['a'].embedding is not None

    with pytest.raises(KeyError, match="'b'"):
        indexer.fill_embedding(DocumentArray([Document(id='b')]))


def test_load(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}
    indexer1 = SimpleIndexer(metas=metas)
    indexer1.index(docs)
    indexer2 = SimpleIndexer(metas=metas)
    assert_document_arrays_equal(indexer2._storage, docs)


def test_index(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}

    # test general/normal case
    indexer = SimpleIndexer(metas=metas)
    indexer.index(docs)
    assert_document_arrays_equal(indexer._storage, docs)

    # test index empty docs
    shutil.rmtree(tmpdir)
    indexer = SimpleIndexer(metas=metas)
    indexer.index(DocumentArray())
    assert not indexer._storage


def test_delete(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = SimpleIndexer(metas=metas)
    indexer.index(docs)
    assert_document_arrays_equal(indexer._storage, docs)

    # delete empty docs
    indexer.delete({})
    assert_document_arrays_equal(indexer._storage, docs)

    # delete first 3 docs
    parameters = {'ids': [f'doc{i}' for i in range(1, 4)]}
    indexer.delete(parameters)
    assert_document_arrays_equal(indexer._storage, docs[3:])

    # delete the rest of the docs stored
    parameters = {'ids': [f'doc{i}' for i in range(4, 7)]}
    indexer.delete(parameters)
    assert not indexer._storage

    # delete from empty storage
    parameters = {'ids': [f'doc{i}' for i in range(4, 7)]}
    indexer.delete(parameters)
    assert not indexer._storage


def test_update(tmpdir, docs, update_docs):
    metas = {'workspace': str(tmpdir)}

    # index docs first
    indexer = SimpleIndexer(metas=metas)
    indexer.index(docs)
    assert_document_arrays_equal(indexer._storage, docs)

    # update first doc
    indexer.update(update_docs)
    assert indexer._storage[0].id == 'doc1'
    assert (indexer._storage[0].embedding == [0, 0, 0, 1]).all()


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_search(tmpdir, metric, docs):
    metas = {'workspace': str(tmpdir)}
    match_args = {'metric': metric}

    # test general/normal case
    indexer = SimpleIndexer(match_args=match_args, metas=metas)
    indexer.index(docs)
    search_docs = deepcopy(docs)
    indexer.search(search_docs)
    for i in range(len(docs)):
        assert search_docs[i].matches[0].id == f'doc{i + 1}'
        assert len(search_docs[i].matches) == len(docs)

    # test search with top_k = 1
    indexer.search(search_docs, parameters={'match_args': {'limit': 1}})
    for i in range(len(docs)):
        assert len(search_docs[i].matches) == 1

    # test search with deafult limit/topk again
    # indexer._match_args should not change as a result of the previous operation
    # so expected length of matches should be the same as the first case
    indexer.search(search_docs)
    for i in range(len(docs)):
        assert len(search_docs[i].matches) == len(docs)

    # test search from empty indexed docs
    shutil.rmtree(tmpdir)
    indexer = SimpleIndexer(metas=metas)
    indexer.index(DocumentArray())
    indexer.search(docs)
    for doc in docs:
        assert not doc.matches

    # test search empty docs
    indexer.search(DocumentArray())
