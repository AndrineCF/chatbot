import time
import asyncio
import uuid
import streamlit as st
from Embeddings import SentenceTransformerEmbeddings
from Database import DBpostgres
from Llm.GroqLlm import GroqLlm
from Retrieval import Retrieval
from Chain import Chain


class App:
    """
    En Streamlit-basert chatbot-applikasjon som integrerer dokumentembeddings, en språkgenereringsmodell,
    og retrieval-basert generering for interaktiv chat.
    """

    def __init__(self,
                 db_path: str,
                 embedding_model: str,
                 model_name: str,
                 collection_name: str,
                 custom_cache_dir: str = None) -> None:
        """
        Initialiserer App-klassen ved å sette opp komponenter for embeddings, database,
        språkmodell og retrieval-augmented generation (RAG).
        """
        self.selected_model_key = None
        self.prompt = None
        self.chat_models = {
            "Llama 3.1": {"name": "llama-3.1-70b-versatile", "temperature": 0.7, "max_tokens": 510},
            "Gemma 2": {"name": "gemma2-9b-it", "temperature": 0.7, "max_tokens": 510},
            "Mistral": {"name": "mixtral-8x7b-32768", "temperature": 0.7, "max_tokens": 510},
        }
        # Bruk eksisterende session_id hvis den finnes i session_state
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        # Initialize components only if they haven't been initialized yet
        if "embedding" not in st.session_state:
            st.session_state.embedding = SentenceTransformerEmbeddings(embedding_model, custom_cache_dir)

        if "db" not in st.session_state:
            st.session_state.db = DBpostgres(db_path, st.session_state.embedding, collection_name)

        if "llm" not in st.session_state:
            st.session_state.llm = GroqLlm(model_name, 0.7, 510)

        if "retrieval" not in st.session_state:
            st.session_state.retrieval = Retrieval(st.session_state.db)

        if "chain_rag" not in st.session_state:
            st.session_state.chain_rag = Chain(st.session_state.llm.get_chat_model(), st.session_state.retrieval.get_retriever())

    async def async_response(self, prompt):
        """Kaller RAG-kjeden asynkront for å generere respons."""
        session_id = st.session_state.session_id

        try:
            response = await asyncio.to_thread(
                st.session_state.chain_rag.chat,  # Kall til Chain.chat()
                prompt,  # Send inn brukerens spørsmål (input)
                session_id=session_id,  # Send session_id som det er
            )
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                response = "Forespørselen er for stor. Prøv å stille kortere spørsmål."
            else:
                response = "Beklager, det oppstod en feil ved behandling av forespørselen."
            print(f"Feil: {e}")
        return response

    def simulate_typing(self, text, delay=0.1):
        """Simulerer skriving av tekst bokstav for bokstav eller ord for ord."""
        # Bruker st.empty() for å oppdatere teksten dynamisk i chatten
        response_container = st.empty()
        response = ""

        # Del opp teksten i ord
        words = text.split()
        for word in words:
            response += word + " "
            # Oppdater teksten etter hvert ord
            response_container.markdown(response)
            time.sleep(delay)
    
    def changeModel(self):
        st.session_state.llm = GroqLlm(self.chat_models[self.selected_model_key]['name'], 
                                       self.chat_models[self.selected_model_key]['temperature'], 
                                       self.chat_models[self.selected_model_key]['max_tokens'])
        st.session_state.chain_rag = Chain(st.session_state.llm.get_chat_model(), st.session_state.retrieval.get_retriever())

    def initialize(self):
        """
        Setter opp Streamlit-brukergrensesnittet for chatboten.
        """
        # Konfigurer Streamlit-sideinnstillinger

        # Opprett sidebar for informasjon
        with st.sidebar:
            st.title("GeoChat")
            st.write("Denne chatboten bruker AI for å besvare spørsmål.")
            st.write("Den har tilgang til diverse ")
            self.selected_model_key = st.radio(
                "**Velg chatmodell**",
                list(self.chat_models.keys()),
                index=0
            )
            self.changeModel()

        # Initialiser samtalehistorikk, og sett session_state variabler
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Vis tidligere meldinger
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Brukerinput via chatboks
        self.prompt = st.chat_input("Melding")

        if self.prompt is not None:
            # Vis brukerens melding
            with st.chat_message("user"):
                st.markdown(self.prompt)
            st.session_state.messages.append({"role": "user", "content": self.prompt})

            # Generer respons asynkront og simulere skriving
            with st.chat_message("assistant"):
                try:
                    #response = asyncio.create_task(self.async_response(self.prompt))
                    session_id = st.session_state.session_id
                    print(session_id)
                    response = st.session_state.chain_rag.chat(self.prompt, session_id)
                    # Simulere at chatboten skriver hvert ord
                    self.simulate_typing(response)

                except Exception as e:
                    response = "Beklager, det oppstod en feil ved behandling av forespørselen."
                    print(f"Feil: {e}")
                    st.markdown(response)

            # Legg til respons i historikk
            st.session_state.messages.append({"role": "assistant", "content": response})
