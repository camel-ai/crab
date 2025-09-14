# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Any, List, Optional, Tuple

from crab import BackendOutput, MessageType
from crab.agents.backend_models.camel_model import CamelModel
from camel.messages import BaseMessage
from langchain.schema import Document 

try:
    from camel.embeddings import OpenAIEmbedding
    from camel.retrievers import VectorRetriever
    from camel.storages import QdrantStorage
    RAG_ENABLED = True
except ImportError:
    RAG_ENABLED = False


class CamelRAGModel(CamelModel):
    def __init__(
        self,
        model: str,
        model_platform: str,
        parameters: dict[str, Any] | None = None,
        history_messages_len: int = 0,
        embedding_model: Optional[str] = "text-embedding-3-small",
        collection_name: str = "knowledge_base",
        vector_storage_path: str = "local_data",
        top_k: int = 3,
        similarity_threshold: float = 0.75,
    ) -> None:
        if not RAG_ENABLED:
            raise ImportError(
                "Please install RAG dependencies: "
                "pip install camel-ai[embeddings,retrievers,storages]"
            )
        
        super().__init__(model, model_platform, parameters, history_messages_len)
        
        self.embedding_model = OpenAIEmbedding() if embedding_model else None
        
        if self.embedding_model:
            self.vector_storage = QdrantStorage(
                vector_dim=self.embedding_model.get_output_dim(),
                path=vector_storage_path,
                collection_name=collection_name,
            )
            self.retriever = VectorRetriever(
                embedding_model=self.embedding_model
            )
        else:
            self.vector_storage = None
            self.retriever = None
            
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def process_documents(self, content_path: str) -> None:
        if not self.retriever or not self.vector_storage:
            raise ValueError("RAG components not initialized")
            
        self.retriever.process(
            content=content_path,
            storage=self.vector_storage, 
        )

    def _enhance_with_context(self, messages: List[Tuple[str, MessageType]]) -> List[Tuple[str, MessageType]]:
        if not self.retriever or not self.vector_storage:
            return messages

        query = next(
            (msg[0] for msg in messages if msg[1] != MessageType.IMAGE_JPG_BASE64),
            ""
        )

        try:
            retrieved_info = self.retriever.query(
                query=query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
            )
        except Exception:
            return messages

        if not retrieved_info:
            return messages

        if not retrieved_info[0].get('payload'):
            return messages

        context = "Relevant context:\n\n"
        for info in retrieved_info:
            context += f"From {info.get('content path', 'unknown')}:\n"
            context += f"{info.get('text', '')}\n\n"

        enhanced_messages = []
        enhanced_messages.append((context, MessageType.TEXT))
        enhanced_messages.extend(messages)

        return enhanced_messages

    def chat(self, messages: List[Tuple[str, MessageType]]) -> BackendOutput:
        enhanced_messages = self._enhance_with_context(messages)
        return super().chat(enhanced_messages)

    def get_relevant_content(self, query: str) -> List[Document]:
        if not self.vector_storage: 
            return []
        
        try:
            return self.retriever.query(
                query=query,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
            )
        except Exception:
            return []
