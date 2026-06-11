import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.hero {
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #334155;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
    margin-top: 20px;
}

.chat-user {
    background: #2563eb;
    color: white;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}

.chat-ai {
    background: #334155;
    color: white;
    padding: 15px;
    border-radius: 15px;
    margin: 10px 0;
}

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📄 AI Resume Analyzer</h1>
    <p>RAG + ATS + JD Matching + Skill Gap Analysis</p>
</div>
""", unsafe_allow_html=True)

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


class BGEEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()

    def embed_query(self, text):
        return embedding_model.encode(text).tolist()


llm = ChatMistralAI(
    model="mistral-small-2506",
    api_key=st.secrets["MISTRAL_API_KEY"]
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant.

Answer ONLY using provided context.

If answer is unavailable,
say "I don't know."

Chat History:
{chat_history}
        """
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}
        """
    )
])

ats_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an ATS Resume Analyzer.

Provide:

1. ATS Score /100
2. Strengths
3. Weaknesses
4. Missing Sections
5. Suggestions
6. Skills

Be strict and realistic.
        """
    ),
    (
        "human",
        """
Resume:
{resume}
        """
    )
])

jd_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Compare resume with job description.

Provide:

1. Match Score /100
2. Matching Skills
3. Missing Skills
4. Strengths
5. Weaknesses
6. Suggestions
        """
    ),
    (
        "human",
        """
Resume:
{resume}

JD:
{jd}
        """
    )
])

skill_gap_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Analyze candidate against JD.

Provide:

1. Matching Skills
2. Missing Skills
3. Weak Areas
4. Learning Priorities
5. Roadmap
6. Recommended Projects
        """
    ),
    (
        "human",
        """
Resume:
{resume}

JD:
{jd}
        """
    )
])

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

if uploaded_file:

    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=BGEEmbeddings(),
    persist_directory="./chroma_db"
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    resume_text = "\n".join(
        [doc.page_content for doc in documents]
    )

    st.sidebar.title("Navigation")

    feature = st.sidebar.radio(
        "Choose Feature",
        [
            "Resume Chat",
            "ATS Analysis",
            "JD Match",
            "Skill Gap Analysis"
        ]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 Resume Pages</h3>
            <h2>{}</h2>
        </div>
        """.format(len(documents)), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧩 Chunks</h3>
            <h2>{}</h2>
        </div>
        """.format(len(chunks)), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model</h3>
            <h2>Mistral</h2>
        </div>
        """, unsafe_allow_html=True)

    if feature == "Resume Chat":

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("💬 Chat With Resume")

        user_question = st.chat_input(
            "Ask something about resume"
        )

        if user_question:

            st.session_state.messages.append(
                HumanMessage(content=user_question)
            )

            docs = retriever.invoke(
                user_question
            )

            context = "\n".join([
                doc.page_content
                for doc in docs
            ])

            history = "\n".join([
                f"Human: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"AI: {msg.content}"
                for msg in st.session_state.messages
            ])

            final_prompt = prompt.invoke({
                "context": context,
                "question": user_question,
                "chat_history": history
            })

            response = llm.invoke(
                final_prompt
            )

            st.session_state.messages.append(
                AIMessage(
                    content=response.content
                )
            )

        for msg in st.session_state.messages:

            if isinstance(msg, HumanMessage):

                st.markdown(
                    f"""
                    <div class="chat-user">
                    👤 {msg.content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="chat-ai">
                    🤖 {msg.content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    elif feature == "ATS Analysis":

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("📊 ATS Resume Analysis")

        if st.button("Analyze Resume"):

            with st.spinner(
                "Analyzing Resume..."
            ):

                final_prompt = ats_prompt.invoke({
                    "resume": resume_text
                })

                response = llm.invoke(
                    final_prompt
                )

                st.success(
                    "Analysis Complete"
                )

                st.write(
                    response.content
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    elif feature == "JD Match":

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Resume vs JD Match")

        jd = st.text_area(
            "Paste Job Description"
        )

        if st.button("Analyze JD"):

            with st.spinner(
                "Matching Resume..."
            ):

                final_prompt = jd_prompt.invoke({
                    "resume": resume_text,
                    "jd": jd
                })

                response = llm.invoke(
                    final_prompt
                )

                st.success(
                    "Matching Complete"
                )

                st.write(
                    response.content
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    elif feature == "Skill Gap Analysis":

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🚀 Skill Gap Analysis"
        )

        jd = st.text_area(
            "Paste Job Description"
        )

        if st.button(
            "Analyze Skills"
        ):

            with st.spinner(
                "Finding Skill Gaps..."
            ):

                final_prompt = skill_gap_prompt.invoke({
                    "resume": resume_text,
                    "jd": jd
                })

                response = llm.invoke(
                    final_prompt
                )

                st.success(
                    "Analysis Complete"
                )

                st.write(
                    response.content
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

