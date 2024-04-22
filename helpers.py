import pickle
import subprocess

import librosa
import loguru
import numpy as np
import zstandard
from matplotlib import pyplot as plt
from loguru import logger

@logger.catch
def visualize(y):
    S = librosa.feature.melspectrogram(y=y)
    mfccs = librosa.feature.mfcc(y=y, n_mfcc=40)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                   x_axis='time', y_axis='mel', fmax=8000,
                                   ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()
@logger.catch
def visualize_features(features):
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    librosa.display.specshow(features, y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='features mfcc')
    ax[0].label_outer()
    plt.show()

@logger.catch
def export_to_MP3(RAWitems, sample_rate, mp3_filename, kbps=160):
    l = subprocess.Popen(f"./bin/lame - -r -s {sample_rate} --preset {kbps} -m m {mp3_filename}", stdin=subprocess.PIPE)
    for item in RAWitems:
        if item.dtype == np.float32:
            max_value = np.iinfo(np.int16).max  # Максимальное значение типа данных int16
            item = np.array(item * max_value, dtype=np.int16)
        l.stdin.write(item)

    # stream.stop_stream()  # останавливаем и закрываем поток
    # stream.close()
    # pa.terminate()  # закрытие записи с микрофона


def model_save(model, file_path):
    # создаем компрессор zstd
    cctx = zstandard.ZstdCompressor()

    # сериализуем модель
    serialized_model = pickle.dumps(model)

    # сжимаем сериализованную модель с помощью компрессора zstd
    compressed_model = cctx.compress(serialized_model)

    # записываем сжатую модель в файл
    with open(file_path, 'wb') as f:
        f.write(compressed_model)
    logger.info(f'Save object {file_path} len={len(serialized_model)}, compressed={len(compressed_model)} ratio = {len(compressed_model)/ len(serialized_model):.3}')
    print(f"The object was saved to file {file_path} using zstd compression")


def model_load(file_path):
    # создаем декомпрессор zstd
    dctx = zstandard.ZstdDecompressor()

    # считываем сжатую модель из файла
    with open(file_path, 'rb') as f:
        compressed_model = f.read()

    # декомпрессируем сжатую модель
    decompressed_model = dctx.decompress(compressed_model)

    # десериализуем модель
    model = pickle.loads(decompressed_model)

    print(f"Object loaded from {file_path} (zstd compression)")
    return model