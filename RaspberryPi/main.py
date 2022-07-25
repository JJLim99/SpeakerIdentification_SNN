import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
import cv2
import os
import numpy as np

tf.compat.v1.disable_v2_behavior()

import struct
import wave
from datetime import datetime
from threading import Thread

import pvporcupine
from pvrecorder import PvRecorder

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QTimer

import time
import sys

class PorcupineDemo(Thread):
    def __init__(
            self,
            access_key,
            library_path,
            model_path,
            keyword_paths,
            sensitivities,
            input_device_index=None,
            output_path=None):

        super(PorcupineDemo, self).__init__()

        self._access_key = access_key
        self._library_path = library_path
        self._model_path = model_path
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities
        self._input_device_index = input_device_index

        self._output_path = output_path

    def run(self):
        keywords = list()
        for x in self._keyword_paths:
            keyword_phrase_part = os.path.basename(x).replace('.ppn', '').split('_')
            if len(keyword_phrase_part) > 6:
                keywords.append(' '.join(keyword_phrase_part[0:-6]))
            else:
                keywords.append(keyword_phrase_part[0])

        porcupine = None
        recorder = None
        wav_file = None
        try:
            porcupine = pvporcupine.create(
                access_key=self._access_key,
                library_path=self._library_path,
                model_path=self._model_path,
                keyword_paths=self._keyword_paths,
                sensitivities=self._sensitivities)

            recorder = PvRecorder(device_index=self._input_device_index, frame_length=porcupine.frame_length)
            recorder.start()

            if self._output_path is not None:
                wav_file = wave.open(self._output_path, "w")
                wav_file.setparams((1, 2, 16000, 512, "NONE", "NONE"))

            print(f'Using device: {recorder.selected_device}')

            print('Listening {')
            for keyword, sensitivity in zip(keywords, self._sensitivities):
                #print('  %s (%.2f)' % (keyword, sensitivity))
                print('  %s ' % (keyword))
            print('}')

            while True:
                pcm = recorder.read()

                if wav_file is not None:
                    wav_file.writeframes(struct.pack("h" * len(pcm), *pcm))

                result = porcupine.process(pcm)

                # Break if wake word is detected
                if result >= 0:
                    print('[%s] Detected %s' % (str(datetime.now()), keywords[result]))
                    return 1

        except KeyboardInterrupt:
            print('Stopping ...')

        finally:
            if porcupine is not None:
                porcupine.delete()

            if recorder is not None:
                recorder.delete()

            if wav_file is not None:
                wav_file.close()

    @classmethod
    def show_audio_devices(cls):
        devices = PvRecorder.get_audio_devices()

        for i in range(len(devices)):
            print(f'index: {i}, device name: {devices[i]}')


def main():
    access_key = "yKmkzwrm60ldOFCx+SvbG43zc40hsr63hvAzYg+s6VUGH+1ZCfQ06A=="    # Change the access_key here
    keywords = "picovoice"
    keywords = keywords.split()

    library_path = pvporcupine.LIBRARY_PATH

    model_path = pvporcupine.MODEL_PATH

    sensitivities=None

    audio_device_index=-1

    output_path="./inference/inspectVoice.wav"

    PorcupineDemo.show_audio_devices()
    
    if access_key is None:
        raise ValueError("AccessKey (--access_key) is required")

    if keywords is None:
        raise ValueError("Either `--keywords` or `--keyword_paths` must be set.")
    keyword_paths = [pvporcupine.KEYWORD_PATHS[x] for x in keywords]

    if sensitivities is None:
        sensitivities = [0.5] * len(keyword_paths)

    if len(keyword_paths) != len(sensitivities):
        raise ValueError('Number of keywords does not match the number of sensitivities.')

    return PorcupineDemo(
        access_key=access_key,
        library_path=library_path,
        model_path=model_path,
        keyword_paths=keyword_paths,
        sensitivities=sensitivities,
        output_path=output_path,
        input_device_index=audio_device_index).run()

model=load_model('./model/MCUNet256kb_Full_64bs_40.hdf5',compile=False)
foldername = "./inference/data/img/"

all_files_list=[]

for file in os.listdir("./inference/data/img/"):
    all_files_list.append(file)

# Sampling frequency
freq = 22050

wavInspectPath = "./inference/inspectVoice.wav"
imgInspectPath = "./inference/inspectVoice.jpg"

def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )
    return contrastive_loss

def create_spectrogram(filepath,save_path):
  plt.interactive(False)
  clip,sample_rate=librosa.load(filepath,sr=None)
  fig=plt.figure(figsize=[0.72,0.72])
  ax=fig.add_subplot(111)
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)
  ax.set_frame_on(False)
  S=librosa.feature.melspectrogram(y=clip,sr=sample_rate)
  librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
  fig.savefig(save_path,dpi=400,bbox_inches='tight',pad_inches=0)
  plt.close()
  fig.clf()
  plt.close(fig)
  plt.close('all')
  del filepath,save_path,clip,sample_rate,fig,ax,S

def load_img(path):
  img=cv2.imread(path)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img=cv2.resize(img,(224,224))
  return img

def match_file(filename):
    score_list=[]
    img1=load_img(filename)
    img1=img1/255
    for i in range(len(all_files_list)):
        img2=load_img('./inference/data/img/'+all_files_list[i])
        img2=img2/255
        X=[np.zeros((1,224,224,3)) for i in range(2)]
        Y=[np.zeros(1,)]
        X[0][0,:,:,:]=img1
        X[1][0,:,:,:]=img2
        score_list.append(model.predict(X))
    score_list=np.array(score_list)
    idx=np.argmax(score_list)
    return all_files_list[idx], score_list[idx], score_list

#Finding all files in a directory.
def num_file(foldername):
    test_list=[]
    files = []

    for r, d, f in os.walk(foldername):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    for f in files:
        test_list.append(f)
    
    return len(test_list)

name,score,score_list=match_file(imgInspectPath)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speaker Identification System")
        self.setGeometry(0, 0, 1960 ,1080)
        self.setStyleSheet("background-color: black;")
        
        self.wakeWord = QLabel("", self)
        self.wakeWord.setGeometry(825, 500, 300, 40)
        self.wakeWord.setFont(QFont("Arial",22))
        self.wakeWord.setStyleSheet("QLabel{background-color: black; color: white; font-weight: bold}")
        
        self.labelRecording = QLabel("", self)
        self.labelRecording.setGeometry(875, 500, 300, 40)
        self.labelRecording.setFont(QFont("Arial",22))
        self.labelRecording.setStyleSheet("QLabel{background-color: black; color: white; font-weight: bold}")
        
        self.labelResult = QLabel("", self)
        self.labelResult.setGeometry(460, 425, 1000, 200)
        self.labelResult.setFont(QFont("Arial",22))
        self.labelResult.setStyleSheet("QLabel{background-color: black; color: white; font-weight: bold}")
        self.labelResult.setAlignment(Qt.AlignCenter)
        
        #self.showFullScreen()
        self.show()
        
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recordStat)
        self.timer.start()

        self.iter = 0

    def recordStat(self):
        self.labelRecording.hide()
        self.labelRecording.repaint()
        self.labelResult.hide()
        self.labelResult.repaint()

        if(self.iter % 2 == 0):
            self.wakeWord.setText("Detecting wake word...")
            self.wakeWord.show()
            self.wakeWord.repaint()
            self.flag = main()
            self.wakeWord.hide()
            self.wakeWord.repaint()

        if(self.flag == 1):
            self.labelRecording.setText("Recording...")
            self.labelRecording.show()
            self.labelRecording.repaint()
            duration = 3
            recording = sd.rec(int(duration * freq),samplerate=freq, channels=2)
            sd.wait()
            #self.labelRecording.setText("Verifying...")
            #self.labelRecording.repaint()
            
            write(wavInspectPath, freq, recording)

            x, fs = librosa.load(wavInspectPath)

            fig=plt.figure(figsize=[0.72,0.72])
            ax=fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

            create_spectrogram(wavInspectPath, imgInspectPath)
            
            start = time.time()
            name,score,score_list=match_file(imgInspectPath)
            end = time.time()
            
            inferenceTime = (end - start) / num_file(foldername)
            
            name = (name.split(".")[0]).split("_")[0]
            self.labelRecording.hide()
            self.labelRecording.repaint()

            if(float(score)*100 >= 90):
                self.labelResult.setText("Authorized user: " + name + "\nInference Time: {:.4f}s".format(inferenceTime) + "\nAccuracy: {:.2f}%".format(float(score)*100))
                print("Authorized user: ", name)
                print("Accuracy: {:.2f}%".format(float(score)*100))
            else:
                self.labelResult.setText("Unrecognized user...\nInference Time: {:.4f}s".format(inferenceTime))
            self.labelResult.show()
            self.labelResult.repaint()
            time.sleep(1)
            
        else:
            self.labelRecording.hide()

        self.iter = self.iter + 1
        self.flag = not self.flag

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
