import pyaudio
import wave
import os
import threading
import time
import warnings

import soundcard as sc
import numpy as np

def get_mp_and_spk_index():
    audio = pyaudio.PyAudio()

    # Print all available microphone devices
    # print("可用的麦克风设备有：")
    # for i in range(audio.get_device_count()):
    #     dev = audio.get_device_info_by_index(i)
    #     if dev.get('maxInputChannels') > 0 and dev.get('hostApi') == 0:
    #         print(f"{i}: {dev.get('name')}")
    #
    # # 选择麦克风设备
    # mp_index = int(input("请选择麦克风设备编号："))
    # device_info = audio.get_device_info_by_index(mp_index)
    # if device_info.get('maxInputChannels') == 0:
    #     raise Exception("选择的设备不支持输入。")

    # Print all available speaker devices
    mics = sc.all_microphones(include_loopback=True)
    loopback_devices = [mic for mic in mics if mic.isloopback]
    loopback_devices_name = [mic.name for mic in loopback_devices]

    if not loopback_devices:
        print("无可用的内录设备!!!")
        return

    print(f"可用的内录设备有：")
    for i, name in enumerate(loopback_devices_name):
        print(f"{i}: {name}")

    lb_idx = input("请选择内录设备编号：")
    try:
        lb_idx = int(lb_idx)
    except ValueError:
        print("输入错误")
        return

    if lb_idx < 0 or lb_idx >= len(loopback_devices):
        print("输入错误")
        return

    # return mp_index, lb_idx
    return 0, lb_idx

def clean_up():
    if os.path.exists("recording"):
        for file in os.listdir("recording"):
            os.remove(os.path.join("recording", file))
            print(f"Deleted {file}")
        print("All recording files have been deleted.")

class MicrophoneRecorder:
    def __init__(self, mp_index=None):
        self.mp_index = mp_index
        self._running = True

    def recording_microphone(self, audio_format=pyaudio.paInt16, channels=1, rate=16000, chunk=1024, filename="speak_record.wav"):
        """
        Record audio from microphone and save as WAV file.
        :param audio_format: 音频格式
        :param channels: 声道数
        :param rate: 采样率
        :param chunk: 缓冲区帧数
        :param record_second: 录音时长
        :param output_fn: 输出文件名
        :return:
        """
        # 初始化PyAudio
        audio = pyaudio.PyAudio()

        # 打开音频流
        stream = audio.open(format=audio_format,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=self.mp_index,
                            frames_per_buffer=chunk)

        print("Starting stream recording from microphone...")

        frames = []

        # 录音
        try:
            while self._running:
                data = stream.read(chunk)
                frames.append(data)
        except KeyboardInterrupt:
            return

        print("Recording finished.")

        # 停止并关闭音频流
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # 保存录音为WAV文件
        self.write_audio(audio, filename, audio_format, channels, rate, frames)


    def write_audio(self, audio, filename, audio_format, channels, rate, frames):
        fp = os.path.join("recording", filename)
        with wave.open(fp, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        print(f"Recording has been saved in {fp}")

    def start(self):
        threading.Thread(target=self.recording_microphone).start()

    def stop(self):
        self._running = False



class LoopbackRecorder:
    def __init__(self, lb_index=None, time_interval=180):
        self.lb_index = lb_index
        self._running = True
        self.time_interval = time_interval

        from soundcard import SoundcardRuntimeWarning
        warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)


    def record_and_save(self, device, filename, samplerate=16000):
        print(f"Starting stream recording from speaker: {device.name}...")

        while self._running:
            recorded_data = []
            start_time = time.time()
            
            with device.recorder(samplerate, 1) as mic:
                try:
                    while self._running and (time.time() - start_time) < self.time_interval:
                        frames = mic.record(numframes=1024)
                        recorded_data.append((frames * 32768).astype(np.int16).tobytes())
                except KeyboardInterrupt:
                    print("\nRecording stopped.")
                    self._running = False

            # Only save the recording if we're still running (stop hasn't been called)
            if self._running and recorded_data:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join("recording", f"{filename}_{timestamp}.wav")
                print(f"Saving recorded data to {filepath}...")
                with wave.open(filepath, mode="wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(samplerate)
                    for data in recorded_data:
                        wav_file.writeframes(data)
                print(f"Recording saved to {filepath}")
            elif not self._running:
                print("Recording stopped. File not saved due to stop request.")


    def __record_loopback(self):
        mics = sc.all_microphones(include_loopback=True)
        loopback_devices = [mic for mic in mics if mic.isloopback]
        loopback_device = loopback_devices[self.lb_index]
        self.record_and_save(loopback_device, filename="loopback_record")


    def start(self):
        # threading._start_new_thread(self.__record_loopback())
        threading.Thread(target=self.__record_loopback).start()

    def stop(self):
        self._running = False


if __name__ == "__main__":
    mp_idx, lb_idx = get_mp_and_spk_index()

    micro_rec = MicrophoneRecorder(mp_idx)
    loop_rec = LoopbackRecorder(lb_idx)

    # 开始录音
    micro_rec.start()
    loop_rec.start()
    # 等待录音
    time.sleep(10)
    # 停止录音，保存文件
    micro_rec.stop()
    loop_rec.stop()
