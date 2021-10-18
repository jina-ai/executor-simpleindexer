from copy import deepcopy
from typing import Dict, Optional

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina.types.arrays.memmap import DocumentArrayMemmap


class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together,
    in a DocumentArrayMemmap object

    To be used as a unified indexer, combining both indexing and searching
    """

    def __init__(
        self,
        match_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initializer function for the simple indexer
        :param match_args: the arguments to `DocumentArray`'s match function
        """
        super().__init__(**kwargs)

        self._match_args = match_args or {}
        self._storage = DocumentArrayMemmap(self.workspace)
        self._embedding_shape = None
        self.logger = JinaLogger('simple_indexer')

    @requests(on='/index')
    def index(
        self,
        docs: Optional['DocumentArray'] = None,
        **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        """
        if not docs:
            return

        for doc in docs:
            if doc.embedding is None:
                self.logger.warning(
                    f'embedding of doc {doc.id} is None, '
                    'skip appending this doc to storage'
                )
                continue
            if (
                self.embedding_shape is not None
                and doc.embedding.shape != self.embedding_shape
            ):
                self.logger.warning(
                    f'embedding shape {doc.embedding.shape} '
                    f'of doc {doc.id} does not match '
                    f'the expected embedding shape {self.embedding_shape}'
                    '(embeddings of all docs in storage should have the same shape), '
                    'skip appending this doc to storage'
                )
                continue
            self._storage.append(doc)

    @requests(on='/search')
    def search(
        self,
        docs: Optional['DocumentArray'] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match function. They overwrite the original match_args arguments.
        """
        if not docs:
            return
        match_args = deepcopy(self._match_args)
        if parameters:
            match_args.update(parameters)
        try:
            docs.match(self._storage, **match_args)
            return
        except ValueError as e:
            self._filter()

        docs.match(self._storage, **match_args)

    def _filter(self):
        if not self._docs:
            return

        doc_id_embeddings = [(doc.id, doc.embedding) for doc in self._storage]
        for _id, embedding in doc_id_embeddings:
            if embedding is None or embedding.shape != self.embedding_shape:
                self.logger.warning(
                    f'filtering storage - embedding of doc {doc.id} is either None '
                    'or has mismatched embedding shape, delete this doc from storage'
                )
                del self._docs[_id]

    @property
    def embedding_shape(self):
        if self._embedding_shape is not None:
            return self._embedding_shape

        for doc in self._storage:
            if doc.embedding is not None:
                self._embedding_shape = doc.embedding.shape
                break
        return self._embedding_shape

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])

        for idx in deleted_ids:
            if idx in self._storage:
                del self._storage[idx]
        # set _embedding_shape to None to update the attribute
        # next time the property is accessed
        self._embedding_shape = None

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], **kwargs):
        """Update doc with the same id, if not present, append into storage

        :param docs: the documents to update
        """

        if not docs:
            return

        for doc in docs:
            if doc.id not in self._storage:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )
                continue
            if doc.embedding is None:
                self.logger.warning(
                    f'embedding of doc {doc.id} is None, skip appending this doc to storage'
                )
                continue
            if (
                self.embedding_shape is not None
                and doc.embedding.shape != self.embedding_shape
            ):
                self.logger.warning(
                    f'embedding shape {doc.embedding.shape} of doc {doc.id} does not match '
                    f'the expected embedding shape {self.embedding_shape}'
                    '(embeddings of all docs in storage should have the same shape), '
                    'skip updating this doc in storage'
                )
                continue
            self._storage[doc.id] = doc

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        if not docs:
            return

        for doc in docs:
            doc.embedding = self._storage[doc.id].embedding
