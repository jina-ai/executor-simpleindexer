import inspect
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
        key_length: int = 64,
        buffer_pool_size: int = 100_000,
        **kwargs,
    ):
        """
        Initializer function for the simple indexer

        To specify storage path, use `workspace` attribute in executor `metas`
        :param match_args: the arguments to `DocumentArray`'s match function
        :param key_length: the `key_length` keyword argument to
        `DocumentArrayMemmap`'s constructor
        :param buffer_pool_size: the `buffer_pool_size` argument to
        `DocumentArrayMemmap`'s constructuor., which stores
            the indexed Documents. During querying, the embeddings of the indexed
            Documents are cached in a buffer pool.
            `buffer_pool_size` sets the number of Documents to be cached. Make sure
            it is larger than the total number
            of indexed Documents to avoid repeating loading embeddings. By default,
            it is set to `100000`.
            Check more information at
            https://docs.jina.ai/api/jina.types.arrays.memmap/?jina.types.arrays
            .memmap.DocumentArrayMemmap.
        """
        super().__init__(**kwargs)

        self._match_args = match_args or {}
        self._storage = DocumentArrayMemmap(
            self.workspace, key_length=key_length, buffer_pool_size=buffer_pool_size
        )
        self.logger = JinaLogger(self.metas.name)

    @requests(on='/index')
    def index(
        self,
        docs: Optional['DocumentArray'] = None,
        **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        """
        if docs:
            self._storage.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: Optional['DocumentArray'] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        if not docs:
            return
        match_args = deepcopy(self._match_args)
        if parameters:
            match_args.update(parameters)

        match_args = SimpleIndexer._filter_parameters(docs, match_args)
        left_trav_path = match_args.pop('traversal_ldarray', None)
        # if it's 'r' we would just be duplicating
        if left_trav_path and left_trav_path != 'r':
            left_trav_docs = DocumentArray(docs.traverse_flat(left_trav_path))
        else:
            left_trav_docs = docs

        right_trav_path = match_args.pop('traversal_rdarray', None)
        # if it's 'r' we would just be duplicating
        if right_trav_path and right_trav_path != 'r':
            right_trav_docs = DocumentArray(
                self._storage.traverse_flat(right_trav_path)
            )
        else:
            right_trav_docs = self._storage

        left_trav_docs.match(right_trav_docs, **match_args)

    @staticmethod
    def _filter_parameters(docs, match_args):
        # get only those arguments that exist in .match
        args = set(inspect.getfullargspec(docs.match).args)
        args.discard('self')
        match_args = {k: v for k, v in match_args.items() if k in args}
        return match_args

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])

        for idx in deleted_ids:
            if idx in self._storage:
                del self._storage[idx]

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], **kwargs):
        """Update doc with the same id, if not present, append into storage

        :param docs: the documents to update
        """

        if not docs:
            return

        for doc in docs:
            try:
                self._storage[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        if not docs:
            return

        for doc in docs:
            doc.embedding = self._storage[doc.id].embedding
