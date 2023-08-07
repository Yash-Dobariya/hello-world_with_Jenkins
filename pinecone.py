import os
from typing import Any

import pinecone
from fastapi import status
from fastapi.exceptions import HTTPException
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    TensorflowHubEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from openai.error import AuthenticationError as OpenAIAuthenticationError

from src.utils.apiserp import scrap_results

from src.config import Config

pinecone.init(
    api_key=Config.PINECONE_API_KEY,
    environment=Config.PINECONE_REGION,
)

DIMENSIONS = {
    "huggingface": 768,
    "cohere": 768,
    "tensorflow": 512,
    "openai": 1536,
}


class PineconeConnector:
    def __init__(
        self,
        dimension,
        chunk_size=300,
        chunk_overlap=200,
        length_function=len,
        metric="cosine",
        embedding_creator="openai",
        default_index_name="default",
        default_name_space="fastapi-bot",
        model="text-embedding-ada-002",
    ) -> None:
        self.model = model
        self.metric = metric
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_name = default_index_name
        self.name_space = default_name_space
        self.length_function = length_function

        if embedding_creator == "cohere":
            self.embeddor = CohereEmbeddings(
                cohere_api_key=os.environ["COHERE_API_KEY"]
            )
        elif embedding_creator == "huggingface":
            self.embeddor = HuggingFaceEmbeddings(
                model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"
            )
        elif embedding_creator == "tensorflow":
            self.embeddor = TensorflowHubEmbeddings()
        else:
            self.embeddor = OpenAIEmbeddings(
                openai_api_key=Config.OPENAI_API_KEY,
                model=self.model,
                disallowed_special=(),
            )

    def create_index(self, index_name: str) -> Any:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name, dimension=self.dimension, metric=self.metric
            )
            return True
        return False

    def create_embedding_from_texts(
        self,
        texts: list,
        index_name: str,
        namespace: str = None,
    ) -> Any:
        try:
            Pinecone.from_texts(
                texts=texts,
                embedding=self.embeddor,
                index_name=index_name,
                namespace=(namespace or self.name_space),
            )

        except OpenAIAuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Something went wrong, please try again shortly",
            )

        return True

    def split_text(self, text: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )
        return text_splitter.split_text(text)

    def create_embeddings(
        self,
        texts: list,
        index_name: str = None,
        name_space: str = None,
    ):
        self.create_index(index_name=(index_name or self.index_name))
        for text in texts:
            self.create_embedding_from_texts(
                texts=self.split_text(text),
                index_name=(index_name or self.index_name),
                namespace=(name_space or self.name_space),
            )

        return True

    def search(
        self,
        query: str,
        top_k: int = 3,
        index_name: str = None,
        namespace: str = None,
    ):
        self.searcher = Pinecone.from_existing_index(
            index_name=(index_name or self.index_name),
            embedding=self.embeddor,
            namespace=(namespace or self.name_space),
        )

        return self.searcher.similarity_search(query, k=top_k)

    def construct_prompt(self, question: str, name_space: str):
        docs = self.search(query=question, namespace=name_space)
        text = ""
        content = []
        for ele in docs:
            content.append(ele.page_content)
        text = " ".join(set(content))
        prompt = f"Act as a helpful AI Assistant. You are SMART an AI based chatbot. \
TASK: Generate a well-detailed Answer \
for the Question based on the Text. Make sure to use the given Text to generate the response\
and structure the response. Carefully follow the TASK and \
give the most relevant response accordingly, Bot response should not include \
any generic information. Make sure that the answer is detailed and descriptive. Be sure to generate an Answer \
that is related to the Text only and it is in your own words. Make \
sure to include any relevant URLs present in the Text that are \
relevant and related to the Question. If the Question is not \
related to the Text, respond with 'Not enough information is \
available at this moment'. Do not add any additional \
information or any suggestions or links in the Answer. if someone greets \
you then greet back in formal way. \n\nText:\n{text}"
        return prompt, text

    def get_answer(self, question: str, include_google_data: bool, name_space: str):

        serpapi_response = scrap_results(question)
        template, docs_data = self.construct_prompt(question, name_space)

        if include_google_data:
            template += f"\nGoogle Data: {serpapi_response}\n"
        template += f"""

Question: {question}
Bot:
        """
        prompt_template = PromptTemplate(template=template, input_variables=[])
        llm_chain = LLMChain(
            llm=OpenAI(
                max_tokens=1000,
                openai_api_key=Config.OPENAI_API_KEY,
                model_name="text-davinci-003",
                temperature=0.7,
            ),
            prompt=prompt_template,
            verbose=True,
        )

        with get_openai_callback() as cb:  # noqa
            result = llm_chain(inputs={})

        return {
            "response": result.get("text"),
            "data_chunk": docs_data,
            "prompt": template,
        }


    def upload_to_pinecone(all_contents, name_space):
        pinecone_connector = PineconeConnector(dimension=DIMENSIONS["openai"])
        pinecone_connector.create_embeddings(all_contents, name_space=name_space)


    def delete_name_space(name_space):
        # Initialize the index with the provided index_name or the default index_name
        index = pinecone.Index(index_name="default")

        delete_response = index.delete(delete_all=True, namespace=name_space)

        return delete_response
