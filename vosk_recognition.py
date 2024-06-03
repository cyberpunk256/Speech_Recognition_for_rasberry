import json  # JSONデータの操作を可能にするライブラリ
import queue  # キューの実装を提供するライブラリ
import sys  # Pythonの実行環境に関する情報を提供するライブラリ
from collections import namedtuple  # 名前付きタプルの作成をサポートするライブラリ

import sounddevice as sd  # オーディオデバイスから音声を取得するためのライブラリ
from vosk import KaldiRecognizer, Model, SetLogLevel  # Voskを使って音声認識を行うためのライブラリ
import speech_recognition as sr  # 音声認識ライブラリ

class MicrophoneStream:
    """マイク音声入力のためのクラス."""

    def __init__(self, rate, chunk):
        """音声入力ストリームを初期化する.
        Args:
           rate (int): サンプリングレート (Hz)
           chunk (int): 音声データを受け取る単位（サンプル数）
        """
        # マイク入力のパラメータ
        self.rate = rate  # サンプリングレート
        self.chunk = chunk  # チャンクサイズ

        # 入力された音声データを保持するデータキュー（バッファ）
        self.buff = queue.Queue()  # キューオブジェクトの生成

        # マイク音声入力の初期化
        self.input_stream = None

    def open_stream(self):
        """マイク音声入力の開始"""
        # マイクからの音声をストリーミングするための設定
        self.input_stream = sd.RawInputStream(
            samplerate=self.rate,  # サンプリングレートの設定
            blocksize=self.chunk,  # ブロックサイズの設定
            dtype="int16",  # サンプルのデータタイプ
            channels=1,  # チャンネル数
            callback=self.callback,  # コールバック関数の設定
        )

    def callback(self, indata, frames, time, status):
        """音声入力の度に呼び出される関数."""
        if status:  # ステータスがある場合、エラーメッセージを表示
            print(status, file=sys.stderr)

        # 入力された音声データをキューへ保存
        self.buff.put(bytes(indata))  # キューにデータを追加

    def generator(self):
        """音声認識に必要な音声データを取得するための関数."""
        while True:  # キューに保存されているデータを全て取り出す
            # 先頭のデータを取得
            chunk = self.buff.get()  # キューからデータを取り出す
            if chunk is None:  # データがない場合、終了
                return
            data = [chunk]  # データリストに追加

            # まだキューにデータが残っていれば全て取得する
            while True:
                try:
                    chunk = self.buff.get(block=False)  # キューからデータを取り出す
                    if chunk is None:  # データがない場合、終了
                        return
                    data.append(chunk)  # データリストに追加
                except queue.Empty:  # キューが空の場合、ループを終了
                    break

            # yieldにすることでキューのデータを随時取得できるようにする
            yield b"".join(data)  # バイトデータを結合して返す

def get_asr_result(vosk_asr):
    """音声認識APIを実行して最終的な認識結果を得る.
    Args:
       vosk_asr (VoskStreamingASR): 音声認識モジュール
    Returns:
       recog_text (str): 音声認識結果
    """
    mic_stream = vosk_asr.microphone_stream  # マイク音声入力のストリームを取得
    mic_stream.open_stream()  # マイク音声入力の開始
    with mic_stream.input_stream:  # マイク音声入力のストリームを使用
        audio_generator = mic_stream.generator()  # オーディオジェネレーターの取得
        for content in audio_generator:  # オーディオジェネレーターからデータを取得
            if vosk_asr.recognizer.AcceptWaveform(content):  # ウェーブフォームの受け入れ
                recog_result = json.loads(vosk_asr.recognizer.Result())  # 認識結果をJSON形式に変換
                recog_text = recog_result["text"].split()  # テキストをスペースで分割
                recog_text = "".join(recog_text)  # 空白記号を除去してテキストを結合
                return recog_text  # 認識結果を返す
        return None  # 認識結果がない場合、Noneを返す

def main(chunk_size=8000):
    """音声認識デモンストレーションを実行.
    Args:
       chunk_size (int): 音声データを受け取る単位（サンプル数）
    """
    SetLogLevel(-1)  # VOSK起動時のログ表示を抑制

    # 入力デバイス情報に基づき、サンプリング周波数の情報を取得
    input_device_info = sd.query_devices(kind="input")  # 入力デバイス情報を取得
    sample_rate = int(input_device_info["default_samplerate"])  # サンプリングレートを取得

    # マイク入力を初期化
    mic_stream = MicrophoneStream(sample_rate, chunk_size)  # マイク音声入力のストリームを初期化

    # 音声認識器を構築
    recognizer = KaldiRecognizer(Model("model"), sample_rate)  # Kaldiモデルとサンプリングレートを使用して認識器を構築

    # マイク入力ストリームおよび音声認識器をまとめて保持
    VoskStreamingASR = namedtuple("VoskStreamingASR", ["microphone_stream", "recognizer"])  # 名前付きタプルを作成
    vosk_asr = VoskStreamingASR(mic_stream, recognizer)  # マイク音声入力ストリームと認識器をまとめて保持

    speech = sr.Recognizer()  # 音声認識器のインスタンスを作成
    while True:  # 無限ループ
        print("＜認識開始＞")  # 認識開始メッセージを出力
        with sr.Microphone() as source:  # マイクからの音声を取得
            speech.adjust_for_ambient_noise(source)  # 周囲のノイズを調整
            try:
                recog_result = get_asr_result(vosk_asr)  # 音声認識結果を取得
            except sr.UnknownValueError:  # 認識できない値の場合
                pass
            except sr.RequestError:  # リクエストエラーが発生した場合
                pass
            except sr.WaitTimeoutError:  # タイムアウトエラーが発生した場合
                pass

        print(f"認識結果: {recog_result}")  # 認識結果を出力
        if recog_result == "終わり":  # 認識結果が"終わり"の場合
            print("Bye")  # 終了メッセージを出力
            break  # ループを抜ける

    print("＜認識終了＞")  # 認識終了メッセージを出力

if __name__ == "__main__":
    main()  # メイン関数の実行
