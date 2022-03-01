import pytest
from jina import Document, DocumentArray, Flow, Executor, requests
from executor import SimpleIndexer
import numpy as np


@pytest.fixture
def docs():
    return DocumentArray([Document(embedding=np.zeros(3)) for _ in range(2)])


class Exec(Executor):
    @requests
    def index(self, docs, **kwargs):
        pass


def test_reload_keep_state(docs, tmp_path):
    metas = {'workspace': str(tmp_path / 'workspace')}

    f = Flow().add(uses=SimpleIndexer, uses_metas=metas)

    with f:
        f.index(docs)
        first_search = f.search(inputs=docs)
        f.post('/status')

    with f:
        f.post('/status')
        second_search = f.search(inputs=docs)
        assert len(first_search[0].matches) == len(second_search[0].matches)
