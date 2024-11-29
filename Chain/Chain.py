from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.retrievers import RetrieverOutputLike
from langchain_core.runnables.history import RunnableWithMessageHistory

from Chain.Documents_chain import setup_stuff_documents_chain
from Document_reader.Read_json import read_json_file

def setup_history_aware_retriever(data,
                                  llm: BaseChatModel,
                                  retriever: BaseRetriever) -> RetrieverOutputLike:
    # Prompt template for contextualizing question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", data["contextualize_q_system_prompt"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    print("Creating history aware retriever")
    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    print("Created history aware retriever")
    return history_aware_retriever


class Chain:
    def __init__(self,
                 llm: BaseChatModel,
                 retriever: BaseRetriever) -> None:

        ### States manage chat history ###
        self.store = {}
        self.conversational_rag_chain = None

        data = read_json_file("instruction.json")

        history_aware_retriever = setup_history_aware_retriever(data, llm, retriever)
        question_answer_chain = setup_stuff_documents_chain(data, llm)


        print("Creating retrieval chain")
        # Chain that combines history-aware retriever with QA chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        print("Created retrieval chain")

        # Final conversational RAG chain with history
        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        print("Created chain")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


    def chat(self, query, session_id=None):
        """
        Generate a response using the conversational RAG chain.
        """
        # Use the provided session ID or fallback to the one from the constructor
        session_id = session_id

        # Add the user's message to history
        history = self.get_session_history(session_id)
        history.add_message(query)
        answer = self.conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )["answer"]
        history.add_ai_message(answer)

        print(history.aget_messages())

        return answer

