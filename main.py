import os
import time
import threading

from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

from dotenv import load_dotenv
from audio_input import get_mp_and_spk_index, clean_up, MicrophoneRecorder, LoopbackRecorder
from audio_recog import AudioRecognition
from kb_build_retrieve import ModelConfig, KnowledgeBaseConfig

load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANG_SMITH_API_KEY")

class State(TypedDict):
    """
    State of the conversation

    microphone: str
        The microphone audio input
    speaker: str
        The speaker audio loopback input
    question: str
        The question asked
    context: List[Document]
        The retrieved context
    answer: str
        The generated answer
        
    """
    microphone: str
    speaker: str

    question: str
    context: List[Document]
    answer: str

class ChatConfig:
    def __init__(self, llm, vec_store):
        self.llm = llm
        self.vector_store = vec_store
        self.ar = AudioRecognition()

    def textualize(self, state: State) -> str:
        # micro_context = self.ar.asr_transcript_by_paraformer("recording/loopback_record.wav")
        max_loopback = max([f for f in os.listdir("recording/") if f.startswith("loopback_record")])
        speaker_context = self.ar.asr_transcript_by_paraformer(f"recording/{max_loopback}")
        return {"speaker": speaker_context}

    def retrieve(self, state: State) -> List[Document]:
        retrieve_prompt = """
        You are an assistant for summary audio input and retrieve the related knowledge context. Use the following pieces of Audio and Question input to retrieve context. If you don't find any related context, just say that you don't know. Keep the answer concise.
        Speak Audio: {speaker}
        Question: {question}
        Answer:
        """
        context_kb = self.vector_store.similarity_search(state["speaker"])
        return {"context": context_kb}

    def generate(self, state: State) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = hub.pull("rlm/rag-prompt")
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}


def record_audio_thread(mp_idx, lb_idx, time_interval):
    """录音线程函数"""
    global is_running
    
    print("🎙️ 录音已开始...")
    # 初始化录音器
    # micro_rec = MicrophoneRecorder(mp_idx)
    loop_rec = LoopbackRecorder(lb_index=lb_idx, time_interval=time_interval)
    
    # 开始录音
    # micro_rec.start()
    
    loop_rec.start()
    
    # 持续录音直到停止信号
    while is_running:
        time.sleep(0.5)
    
    # 停止录音，保存文件
    # micro_rec.stop()
    loop_rec.stop()
    print("🎙️ 录音已停止")


def analyze_audio_thread(embeddings, chroma_ef, llms, time_interval):
    """音频分析线程函数"""
    global is_running
    
    # 初始化知识库
    kb = KnowledgeBaseConfig(chroma_ef, embeddings)
    vec_store = kb.get_or_create_kb("kb_files/优势谈判：适用于任何场景的经典谈判.pdf", doc_size=1000, doc_overlap=100)
    
    # 初始化聊天配置
    kb_chat = ChatConfig(llms, vec_store)
    
    while is_running:
        # 等待指定时间间隔
        time.sleep(time_interval+3)
        
        if not is_running:
            break
            
        print("\n🧠 分析线程已启动...")
        
        # 构建并执行图
        graph_builder = StateGraph(State).add_sequence([kb_chat.textualize, kb_chat.retrieve, kb_chat.generate])
        graph_builder.add_edge(START, "textualize")
        graph = graph_builder.compile()

        for step in graph.stream(
                {"question": "Please show me the meeting key points"}, stream_mode="updates"
        ):
            print(f"{step}\n\n----------------\n")


def input_listener_thread():
    """输入监听线程函数"""
    global is_running
    print("\n🛑 输入 'stop' 并按回车终止程序...")
    while True:
        user_input = input().strip().lower()
        if user_input == 'stop':
            is_running = False
            print("⏳ 正在停止程序...")
            break


if __name__ == "__main__":
    ###============== 初始化区 ==============###
    all_model_config = ModelConfig(api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_BASE_URL"))
    embeddings = all_model_config.embedding_model()
    chroma_ef = all_model_config.chroma_ef()
    llms = all_model_config.llm_model()

    mp_idx, lb_idx = get_mp_and_spk_index()
    time_interval = 60 # 每次录音时长（秒）

    is_running = True
    
    ###============== 创建并启动线程 ==============###
    # 创建线程
    record_thread = threading.Thread(target=record_audio_thread, args=(mp_idx, lb_idx, time_interval))
    analyze_thread = threading.Thread(target=analyze_audio_thread, args=(embeddings, chroma_ef, llms, time_interval))
    input_thread = threading.Thread(target=input_listener_thread)
    
    # 设置为守护线程，这样主程序退出时它们也会退出
    record_thread.daemon = True
    analyze_thread.daemon = True
    
    # 启动线程
    record_thread.start()
    analyze_thread.start()
    input_thread.start()
    
    # 等待输入线程完成（即用户输入stop）
    input_thread.join()
    
    # 等待其他线程完成
    print("正在等待线程结束...")
    time.sleep(2)  # 给其他线程一些时间来优雅地结束
    
    ###============== 清理缓存区 ==============###
    clean_up()
    print("程序已结束")