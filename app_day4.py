import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import time

st.set_page_config(
    page_title="Research Knowledge Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research Knowledge Assistant")
st.write("RAG powered Q&A over Tsegai's research knowledge base")
st.divider()

# Knowledge base documents
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Federated Learning for Manufacturing Defect Detection",
        "content": """Federated learning is a machine learning approach that trains 
        models across multiple decentralized devices or servers holding local data 
        samples, without exchanging the raw data. In manufacturing quality control, 
        federated learning enables multiple production sites to collaboratively train 
        defect detection models while keeping sensitive production data private. Each 
        site trains a local model and only shares model updates, not raw images or 
        sensor data. This approach is particularly valuable in additive manufacturing 
        where multiple facilities need to detect defects in 3D printed composite 
        materials."""
    },
    {
        "id": "doc2",
        "title": "Differential Privacy in Machine Learning",
        "content": """Differential privacy is a mathematical framework for protecting 
        individual privacy in datasets. In machine learning, differential privacy adds 
        carefully calibrated noise to model updates or training data to prevent the 
        model from memorizing sensitive information. The privacy budget epsilon controls 
        the tradeoff between privacy protection and model accuracy. Lower epsilon means 
        stronger privacy but potentially lower accuracy. Differential privacy is 
        essential in healthcare AI, financial modeling, and any domain where data 
        contains sensitive personal information."""
    },
    {
        "id": "doc3",
        "title": "UAV Route Optimization and Autonomous Navigation",
        "content": """Autonomous UAV systems require sophisticated path planning 
        algorithms to navigate complex environments safely and efficiently. Route 
        optimization for UAVs involves minimizing flight time and energy consumption 
        while avoiding obstacles and adhering to airspace restrictions. Key algorithms 
        include A-star for grid-based planning, RRT for continuous space planning, and 
        reinforcement learning for adaptive navigation. Sensor fusion combines GPS, IMU, 
        camera, and LiDAR data to build a real-time world model for autonomous decision 
        making. The SkyCare UAV system integrates these capabilities for autonomous 
        perimeter security and monitoring applications."""
    },
    {
        "id": "doc4",
        "title": "Airline Delay Propagation Forecasting",
        "content": """Flight delay propagation is a critical challenge in airline 
        network management. When a flight is delayed, the delay propagates through the 
        network affecting connecting flights and passengers. Hybrid RNN models combining 
        LSTM and GRU layers effectively capture both long-range dependencies across the 
        flight network and short-term propagation patterns. The Copa Airlines study 
        analyzed 75,000 flight records to forecast delay propagation. The model achieved 
        significant improvement over baseline approaches by learning temporal patterns 
        in delay cascades. This research was published in the Journal of Financial 
        Mathematics."""
    },
    {
        "id": "doc5",
        "title": "Ontology-Based Attribute Learning for Zero-Shot Defect Detection",
        "content": """Ontology-based attribute learning enables machine learning models 
        to identify defect types they have never seen during training. This approach 
        mimics human cognitive learning where new instances are recognized through their 
        attributes rather than memorized examples. The system combines BERT for semantic 
        text encoding of defect attribute descriptions with ResNet-50 as a visual 
        backbone for image feature extraction. An ontology framework maps visual 
        evidence to semantic attribute descriptions enabling zero-shot generalization. 
        This work was published in IEEE Transactions on Automation Science and 
        Engineering and resulted in a patent filing."""
    },
    {
        "id": "doc6",
        "title": "Intelligent Transportation Systems and Optimization",
        "content": """Intelligent transportation systems use AI and optimization to 
        improve traffic flow, reduce congestion, and enhance safety. Operations research 
        methods including linear programming, stochastic optimization, and simulation 
        are applied to transportation network design and management. The R-SEAT Center 
        at FAMU-FSU conducts USDOT-funded research on autonomous mobility, connected 
        vehicles, and multimodal transportation optimization. Key research areas include 
        last-mile logistics, truck-drone integrated delivery systems, and mobility 
        analytics with differential privacy preservation."""
    },
    {
        "id": "doc7",
        "title": "Vehicle Sales Forecasting with Machine Learning",
        "content": """Accurate vehicle sales forecasting is critical for automotive 
        industry planning including inventory management, pricing strategy, and 
        production scheduling. At GM Financial, machine learning models were developed 
        to forecast vehicle sales using structured financial datasets containing over 
        109 million records. The pipeline used Python, SAS, and SQL for data processing 
        with Kubeflow and MinIO for scalable model deployment. Feature engineering 
        extracted temporal patterns, seasonal trends, and economic indicators. Models 
        were validated using out-of-sample testing and deployed to support live pricing 
        and inventory decisions."""
    },
    {
        "id": "doc8",
        "title": "Transfer Learning and BERT for NLP Tasks",
        "content": """Transfer learning leverages knowledge from pre-trained models to 
        improve performance on new tasks with limited data. BERT pre-trained on Wikipedia 
        and BookCorpus captures deep contextual language representations. Fine-tuning 
        BERT on domain specific datasets achieves state-of-the-art performance on text 
        classification, named entity recognition, and question answering tasks. The key 
        advantage is that BERT learns general language understanding during pre-training, 
        then adapts to specific tasks with relatively small labeled datasets during 
        fine-tuning."""
    },
    {
        "id": "doc9",
        "title": "Computer Vision for Manufacturing Quality Control",
        "content": """Computer vision systems enable automated quality inspection in 
        manufacturing by analyzing images and video from production line cameras. 
        Convolutional neural networks detect surface defects, dimensional errors, and 
        assembly faults with high accuracy. Real-time inference systems process camera 
        feeds at production line speeds, flagging defects before parts move downstream. 
        Key challenges include class imbalance since defects are rare compared to normal 
        parts, varying lighting conditions, and the need to detect novel defect types 
        not seen during training. Zero-shot learning approaches address the novel defect 
        challenge using attribute-based classification."""
    },
    {
        "id": "doc10",
        "title": "MLOps and Production Machine Learning Pipelines",
        "content": """MLOps combines machine learning, DevOps, and data engineering to 
        deploy and maintain ML models in production reliably. Key components include 
        experiment tracking with MLflow, pipeline orchestration with Kubeflow, model 
        versioning and registry, automated testing, CI/CD for model deployment, and 
        monitoring for data drift and model degradation. At GM Financial, Kubeflow and 
        MinIO were used to build scalable ML pipelines processing millions of records 
        for vehicle sales forecasting. Production models require continuous monitoring 
        to detect when data distributions shift and model performance degrades."""
    }
]

@st.cache_resource
def initialize_rag():
    # Load embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create ChromaDB collection
    client = chromadb.Client()
    
    collection = client.create_collection(
        name="research_kb",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents
    texts = [doc["content"] for doc in DOCUMENTS]
    ids = [doc["id"] for doc in DOCUMENTS]
    titles = [doc["title"] for doc in DOCUMENTS]
    embeddings = embedding_model.encode(texts).tolist()
    
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids,
        metadatas=[{"title": title} for title in titles]
    )
    
    return embedding_model, collection

# Load RAG system
with st.spinner("Initializing RAG system..."):
    embedding_model, collection = initialize_rag()
st.success(f"Knowledge base ready with {collection.count()} documents")

# Sidebar
st.sidebar.header("Settings")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_..."
)

n_docs = st.sidebar.slider(
    "Documents to retrieve",
    min_value=1,
    max_value=5,
    value=3,
    help="How many documents to use as context"
)

show_sources = st.sidebar.checkbox("Show source documents", value=True)
show_similarity = st.sidebar.checkbox("Show similarity scores", value=True)

st.sidebar.divider()
st.sidebar.subheader("Example Questions")
example_questions = [
    "How does federated learning protect privacy?",
    "What algorithms are used for UAV navigation?",
    "How was the GM Financial ML pipeline built?",
    "What is zero-shot defect detection?",
    "How do airline delays propagate through networks?",
    "What is differential privacy epsilon?",
    "How does BERT transfer learning work?",
    "What MLOps tools were used in production?"
]

for q in example_questions:
    if st.sidebar.button(q, key=q):
        st.session_state.question = q

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Ask a question about the research",
        value=st.session_state.get("question", ""),
        placeholder="e.g. How does federated learning protect privacy in manufacturing?",
        height=100
    )

with col2:
    st.markdown("**Knowledge Base Topics:**")
    topics = [
        "🔒 Federated Learning",
        "🔐 Differential Privacy",
        "🚁 UAV Navigation",
        "✈️ Airline Delays",
        "🔍 Defect Detection",
        "🚗 Transportation AI",
        "📈 Sales Forecasting",
        "📝 BERT and NLP",
        "👁️ Computer Vision",
        "⚙️ MLOps"
    ]
    for topic in topics:
        st.write(topic)

if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not api_key.strip():
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Searching knowledge base and generating answer..."):
            start_time = time.time()

            # Retrieve documents
            query_embedding = embedding_model.encode([question]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_docs,
                include=["documents", "metadatas", "distances"]
            )

            retrieved_docs = []
            for i in range(len(results["ids"][0])):
                title = results["metadatas"][0][i]["title"]
                content = results["documents"][0][i]
                similarity = 1 - results["distances"][0][i]
                retrieved_docs.append({
                    "title": title,
                    "content": content,
                    "similarity": similarity
                })

            # Build context
            context = ""
            for i, doc in enumerate(retrieved_docs):
                context += f"\nDocument {i+1}: {doc['title']}\n{doc['content']}\n"

            # Build prompt
            prompt = f"""You are a helpful AI research assistant for Tsegai Yhdego, 
a PhD researcher in Industrial Engineering specializing in AI and ML.

Answer the question based ONLY on the provided context documents. 
If the answer is not in the context, clearly state that.
Be specific and technical in your response.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            # Generate answer
            groq_client = Groq(api_key=api_key)
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )

            answer = response.choices[0].message.content
            elapsed = (time.time() - start_time) * 1000

        # Display answer
        st.divider()
        st.subheader("Answer")
        st.markdown(answer)

        # Metrics
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Response Time", f"{elapsed:.0f}ms")
        with col_b:
            st.metric("Documents Used", len(retrieved_docs))
        with col_c:
            top_similarity = retrieved_docs[0]["similarity"] if retrieved_docs else 0
            st.metric("Top Similarity", f"{top_similarity:.2%}")

        # Sources
        if show_sources:
            st.divider()
            st.subheader("Source Documents")
            for i, doc in enumerate(retrieved_docs):
                with st.expander(
                    f"Source {i+1}: {doc['title']}" +
                    (f" — {doc['similarity']:.2%} similar" if show_similarity else "")
                ):
                    st.write(doc["content"])

# Conversation history
st.divider()
st.subheader("Browse Knowledge Base")

selected_doc = st.selectbox(
    "Select a document to read",
    options=[doc["title"] for doc in DOCUMENTS]
)

selected_content = next(
    doc["content"] for doc in DOCUMENTS
    if doc["title"] == selected_doc
)

st.markdown(f"**{selected_doc}**")
st.write(selected_content)

st.divider()
st.caption("RAG Pipeline — ChromaDB + Sentence Transformers + Groq LLaMA | Built by Tsegai Yhdego — PhD Industrial Engineering | AI/ML Researcher")
