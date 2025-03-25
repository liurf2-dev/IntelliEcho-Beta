import dashscope
import os

from dotenv import load_dotenv
from http import HTTPStatus
from dashscope.audio.asr import Recognition
from dashscope import MultiModalConversation

load_dotenv()
dashscope.api_key = os.getenv("QWEN_API_KEY")

class AudioRecognition:
    def __init__(self, model="qwen-audio-asr"):
        self.model = model

    def asr_transcript_by_qwen(self, wav_file):
        messages = [
            {
                "role": "user",
                "content": [{"audio": wav_file}],
            }
        ]

        response = MultiModalConversation.call(model=self.model, messages=messages)
        ttl_st = response['output']['choices'][0]['message']['content'][0]['text']
        print("识别结果：", ttl_st) # 缺少标点符号
        return ttl_st

    def asr_transcript_by_paraformer(self, wav_file):
        recognition = Recognition(model='paraformer-realtime-v2',
                                  format='wav',
                                  sample_rate=16000,
                                  # “language_hints”只支持paraformer-v2和paraformer-realtime-v2模型
                                  language_hints=['zh', 'en'],
                                  callback=None)
        result = recognition.call(wav_file)
        if result.status_code == HTTPStatus.OK:
            print('识别结果：')
            ttl_st = ''
            for sentence in result.get_sentence():
                st = sentence['text']
                ttl_st += st
            print(ttl_st)
            return ttl_st
        else:
            print('Error: ', result.message)


# 执行识别和输出
if __name__ == "__main__":
    # WAV文件路径
    wav_file = "recording/loopback_record.wav"
    ar = AudioRecognition()
    ar.asr_transcript_by_qwen(wav_file)