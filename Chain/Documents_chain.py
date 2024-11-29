from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import RetrieverOutputLike

def setup_stuff_documents_chain(data,
                                llm: BaseChatModel) -> RetrieverOutputLike:
    # QA Prompt template
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", data["system_prompt"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    print("Creating question answer chain")
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    print("Created question answer chain")
    return question_answer_chain
