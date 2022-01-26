import inspect
from copy import deepcopy
from typing import Dict, Optional
import os
import uuid

from jina import DocumentArray, Document, Executor, requests
from jina.logging.logger import JinaLogger



class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together in a DocumentArray,
    and can dump to and load from disk.

    To be used as a unified indexer, combining both indexing and searching
    """

    FILE_NAME = 'index.bin'

    def __init__(
        self,
        match_args: Optional[Dict] = None,
        protocol: str = 'pickle-array',
        compress: str = None,
        **kwargs,
    ):
        """
        Initializer function for the simple indexer

        To specify storage path, use `workspace` attribute in executor `metas`
        :param match_args: the arguments to `DocumentArray`'s match function
        :param key_length: the `key_length` keyword argument to
        `DocumentArrayMemmap`'s constructor
        :param protocol: serialisation protocol for disk access
        :param compress: compression algorithm for disk access
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

        self.protocol = protocol
        self.compress = compress
        self._match_args = match_args or {}
        self._index = DocumentArray()
        self.logger = JinaLogger(self.metas.name)
        try:
            self.load_from_disk()
        except FileNotFoundError:
            self.logger.info(
                f'no data found in the workspace'
            )

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
            self._index.extend(docs)

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
        match_args = {**self._match_args, **parameters} if parameters is not None else self._match_args
        match_args = SimpleIndexer._filter_match_params(docs, match_args)
        docs.match(self._index, **match_args)

    @staticmethod
    def _filter_match_params(docs, match_args):
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
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], **kwargs):
        """Update doc with the same id, if not present, append into storage

        :param docs: the documents to update
        """

        if not docs:
            return

        for doc in docs:
            try:
                self._index[doc.id] = doc
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
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/dump')
    def dump(self, docs: Optional[DocumentArray]=None, **kwargs):
        """dump indexed Documents to disk
        """
        if docs:
            self._index.extend(docs)
        bytes = self._index.to_bytes(protocol=self.protocol, compress=self.compress)
        with open(os.path.join(self.workspace, SimpleIndexer.FILE_NAME), 'wb') as f:
            f.write(bytes)

    @requests(on='/load')
    def load_from_disk(self, docs: Optional[DocumentArray]=None, **kwargs):
        with open(os.path.join(self.workspace, SimpleIndexer.FILE_NAME), 'rb') as f:
            self._index = DocumentArray.from_bytes(f, protocol=self.protocol, compress=self.compress)
        if docs:
            self._index.extend(docs)

