from gtts import gTTS
import pygame
import os


def ヘボン音声生成(テキスト, ファイル名):
    # TTSオブジェクトを作成し、言語を日本語に指定する
    tts = gTTS(text=テキスト, lang='ja')
    # オーディオファイルとして保存
    tts.save(ファイル名)


def サウンド再生(ファイルパス):
    # pygameオーディオモジュールを初期化
    pygame.mixer.init()
    # サウンドファイルをロード
    sound = pygame.mixer.Sound(ファイルパス)
    # サウンドを再生
    sound.play()
    # サウンドが再生し終わるまで待つ
    pygame.time.wait(int(sound.get_length() * 1000))  # ミリ秒に変換するために1000を掛ける


# サンプルの日本語テキスト
日本語テキスト = "ターゲット検出"

# 日本語のオーディオを生成して再生する
ヘボン音声生成(日本語テキスト, "japanese_audio.mp4")
サウンド再生("japanese_audio.mp4")

# 生成したファイルを削除（オプション）
# os.remove("japanese_audio.mp3")
