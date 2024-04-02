import subprocess

import pyaudio
import numpy as np

from loguru import logger
import sys
logger.remove()
logger.add("alarm.log", rotation="21 MB", retention = 3, compression="zip", backtrace=True, diagnose=True)  # Automatically rotate too big file
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>", level='INFO')


# частота дискретизации
RATE=22050
# Количество кадров звука в каждой итерации
CHUNK_SIZE = 1*RATE

# Инициализация PyAudio
pa = pyaudio.PyAudio()

# доступные устройства
for i in range(pa.get_device_count()):
    print(i, pa.get_device_info_by_index(i)['name'])

# Открытие потока для захвата звука с микрофона
stream = pa.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

record_seconds=16 # как долго будем писать?

# Захват звука с микрофона и запись 

l = subprocess.Popen(f"./bin/lame - -r -s {RATE} --preset 160 -m m recorded.mp3", stdin=subprocess.PIPE)
for i in range(int(RATE / CHUNK_SIZE * record_seconds)):
        item=stream.read(CHUNK_SIZE)
        l.stdin.write(item)

stream.stop_stream() # останавливаем и закрываем поток 
stream.close()
pa.terminate()