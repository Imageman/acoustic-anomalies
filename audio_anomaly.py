import buffer

import pyaudio
import numpy as np
import librosa

import helpers

try:
    import platform
    if float(platform.release()) > 7:  # Windows 7 
        print(f'Winver {platform.release()}')
        from sklearnex import patch_sklearn  # https://intel.github.io/scikit-learn-intelex/
        patch_sklearn()  # Intel заплатка для ускорения sklearn
    else:
        print('Minimum version for Intel sklearnex Win8.')
except:
    print('For faster work sklearn install sklearnex')
    pass

from helpers import export_to_MP3, model_save, model_load

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tqdm import trange
import argparse
import os.path
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA

from loguru import logger
import sys

logger.remove()
logger.add("audio_anomaly.log", rotation="10 MB", retention=3, compression="zip", backtrace=True,
           diagnose=True)  # Automatically rotate too big file
try:
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> <level>{message}</level>", level='INFO')
except  Exception as e:
    logger.debug(f'logger.add(sys.stdout) Error: {str(e)}')

parser = argparse.ArgumentParser(description='Audio anomaly detect and save .mp3')
parser.add_argument('--learning_time', '-l', default=5, type=int,
                    help=r'Learning time (in minutes) for normal sound profile create. (Default: 5 minutes.)')
parser.add_argument('--n_mfcc', '-n', default=25, type=int,
                    help=r'Number of MFCCs  (10-3500). (Default: 25)')
parser.add_argument('--num_components', '-reduce_features', default=1000, type=int,
                    help=r'Principal Component Analysis components count (10-3500). (Default: 1000; 0 - disabled.)')
parser.add_argument('--num_estimators', '-e', default=300, type=int,
                    help=r'Number of simple evaluators for anomaly detection. More is better. (10-3500). (Default: 300)')
parser.add_argument('--contamination', '-c', default=0.005, type=float,
                    help=r'Expected ratio of anomalies in the training sample. The more noisy the sample is, the higher the number should be set (0.0-0.5). (Default: 0.005)')
parser.add_argument('--MP3_len', '-mp3', default=14, type=int,
                    help=r'Length of .mp3 file. (Default: 14 seconds.)')
parser.add_argument('--model_name', '-m', default='silence',
                    help=r'Name of normal sound profile. (Default: silence)')
parser.add_argument('--save_path', '-o', default='./anomaly_mp3/',
                    help=r'Save mp3 files. (Default: ./anomaly_mp3/)')

args = parser.parse_args()

RATE = 22050  # частота дискретизации
CHUNK_SIZE = 1 * RATE  # Количество кадров звука в каждой итерации
SECONDS_MP3 = args.MP3_len  # длина записанного файла
FOREST_FRAMES_INPUT = 8 * 44  # 44 фрейма это примерно 1 сек
learning_signal_left_count = args.learning_time * 60  # как долго будем собирать статистику?
MP3PATH=args.save_path
logger.info(f'The {FOREST_FRAMES_INPUT//44} sec. of recording is used for analysis. \nRecording to {MP3PATH}')

# FLOAT_TYPE = np.float16
FLOAT_TYPE = np.float32


# Извлечение признаков из звуковых данных
def extract_features(audio_data, n_mfcc):
    spectr = librosa.stft(audio_data)  # Извлечение спектрограммы из звуковых данных
    # Извлечение Mel-спектрограммы из спектрограммы
    mel = librosa.feature.melspectrogram(S=spectr)
    # Извлечение MFCC из Mel-спектрограммы
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(np.abs(mel) ** 2), n_mfcc=n_mfcc)

    # return np.mean(mfcc, axis=1) # Возвращение средних значений MFCC по времени

    return mfcc  # возвращение всей ленты


def get_line(data_array: list, num='last'):
    """
    из 2D массива исторических данных получаем длинный массив
    :param data_array:
    :param num:
    :return:
    """
    if isinstance(num, str) and num == 'last':
        index = len(data_array) - FOREST_FRAMES_INPUT
    else:
        index = num
    result = np.empty(0, dtype=FLOAT_TYPE)
    for i in range(index, index + FOREST_FRAMES_INPUT):
        result = np.concatenate((result, data_array[i].astype(FLOAT_TYPE)), axis=None)
    return result.reshape(1, result.shape[0])




class AudioPCAModel:
    """
    A very fast model. PCA to reduce dimensionality if necessary, and then the the isolation forest reveals anomalies.

    Очень быстрая модель. При необходимости сделается PCA для сокращения размерности, а потом
    изолирующий лес выявляет аномалии
    """
    def __init__(self, num_components,contamination=0.005, n_estimators=300):
        self.num_components = num_components
        self.contamination=contamination
        self.model = None
        self.reduce_features = None
        self.n_estimators=n_estimators
        if self.contamination<0:
            logger.warning(f'contamination={self.contamination}, replace with 0.005')
            self.contamination=0.005
        if self.contamination>0.5:
            logger.warning(f'contamination={self.contamination}, replace with 0.5')
            self.contamination=0.5
        if self.n_estimators<30:
            logger.warning(f'n_estimators={n_estimators}, replace with 30')
            self.n_estimators = 30
        if self.n_estimators<300:
            logger.info(f'n_estimators={n_estimators}. The number of estimators is small. Increase to 300 or more.')

    def learn_model(self, raw_buffer):
        logger.info('Learning AudioPCAModel: Start analyse data build')
        test = get_line(raw_buffer.items, 0)
        X_train = np.empty((len(raw_buffer) - FOREST_FRAMES_INPUT, test.shape[1]), dtype=FLOAT_TYPE)
        logger.info(f'Dimensionality of the training set {X_train.shape}')
        # X_train=X_train.reshape(1, X_train.shape[0])
        for i in trange(0, len(raw_buffer) - FOREST_FRAMES_INPUT):
            X_train[i] = get_line(raw_buffer.items, i)[0]
        if self.num_components > 0:
            # proborFA(X_train)
            logger.info('Start reduce_features fit')
            # self.reduce_features = FactorAnalysis(n_components=self.num_components)
            self.reduce_features = PCA(n_components=self.num_components)
            self.reduce_features.fit(X_train)
            logger.info('Start reduce_features transform')
            X_train = self.reduce_features.transform(X_train)
            logger.info(f'Dimensionality of the training set after PCA reduce_features {X_train.shape}')
        logger.info('Start IsolationForest fit')
        logger.debug(f'n_estimators={self.n_estimators}')
        self.model = IsolationForest(n_estimators=self.n_estimators, max_samples=0.9, contamination=self.contamination, verbose=1)
        self.model.fit(X_train)
        logger.info('End learning')
        return self.model

    def predict(self, line):
        if self.num_components > 0:
            reduced_line = self.reduce_features.transform(line)
            return self.model.predict(reduced_line)
        else:
            return self.model.predict(line)


logger.info('Start')

if not os.path.isdir('bin') or not os.path.isfile('./bin/lame.exe'):
    logger.warning('Create folder bin, download and put binary file lame.exe')

model = None

history_buff = buffer.RecentItems((SECONDS_MP3 * RATE) // CHUNK_SIZE + 1)
mfcc_buff = buffer.RecentItems((RATE * 2) // 1000 * learning_signal_left_count)

# Инициализация PyAudio
pa = pyaudio.PyAudio()

# доступные устройства
# for i in range(pa.get_device_count()):
#    print(i, pa.get_device_info_by_index(i)['name'])

# Открытие потока для захвата звука с микрофона
stream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

# Создание объекта для отслеживания аномалий по величине дисперсии
scaler = StandardScaler()

need_to_save_mp3 = -1  # признак того, что там нужно записать в файл весь буфер, когда переменная равна 0

if os.path.isfile(f'./model/{args.model_name}.zstd'): # загрузка стандартной модели
    logger.info(f'Load model {args.model_name}.zstd')
    model = model_load(f'./model/{args.model_name}.zstd')
if os.path.isfile(f'./model/{args.model_name}_n.zstd'): # загрузка экспериментальной модели
    logger.info(f'Load experimental model {args.model_name}_n.zstd')
    model_n = model_load(f'./model/{args.model_name}_n.zstd')
    model_n.init_codename()

smoothed_score = 0

# DEBUG
# mfcc_buff = model_load('./model/home_silence_mfcc_buff.zstd')
# learning_signal_left_count = 8
# model = AudioPCAModel(args.num_components)
# model.learn_model(mfcc_buff)
# model_n = AudioNormsModel('silence')
# model_n.learn_model(mfcc_buff.items)
# model_save(model_n, f'./model/{args.model_name}_n.zstd')

# Захват звука с микрофона и извлечение признаков в реальном времени
while True:
    # Чтение кадров звуковых данных из потока
    audio_data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.float32)
    history_buff.add_item(audio_data)
    # Извлечение признаков из звуковых данных
    features = extract_features(audio_data, n_mfcc=args.n_mfcc)

    # helpers.visualize_features(features)
    features = features.transpose()
    for line in features:
        mfcc_buff.add_item(line)

    if learning_signal_left_count > 0:
        scaler.partial_fit(features)
        learning_signal_left_count -= 1
        if learning_signal_left_count == 0: logger.info(f'\nSignal statistics:\nmean {scaler.mean_}\nscale {scaler.scale_} ')

    scaled_data_chunk = scaler.transform(features)
    # score = np.mean(np.abs(scaled_data_chunk))/1.1
    signal_str = f'Anomaly: {(np.percentile(np.abs(scaled_data_chunk),90)-0.95)/1.1:.3}'
    logger.debug(signal_str)
    score = (np.percentile(np.abs(scaled_data_chunk), 90) - 0.95)/1.2
    if smoothed_score>0.4:
        signal_str = f'Anomaly smoothed_score={smoothed_score:.3}'
        if score>1.5: signal_str += f'[90% {score:.3}]'


    if learning_signal_left_count < 1 and model == None:
        logger.info('Creating a model for normal sounds.')
        # model_n = AudioNormsModel('silence') # экспериментальная модель
        # model_n.learn_model(mfcc_buff.items)
        # model_save(model_n, f'.\model\{args.model_name}_n.zstd')
        # model_n = model_load(f'.\model\{args.model_name}_n.zstd')

        model = AudioPCAModel(args.num_components, n_estimators= args.num_estimators, contamination=args.contamination)
        model.learn_model(mfcc_buff)
        model_save(model, f'.\model\{args.model_name}.zstd')
        model_save(mfcc_buff, f'.\model\{args.model_name}_mfcc_buff.zstd')
        model = model_load(f'.\model\{args.model_name}.zstd')
        logger.debug(f'We reduce the size of mfcc_buff from {mfcc_buff.max_items} to {FOREST_FRAMES_INPUT * 3} after training')
        mfcc_buff.max_items = FOREST_FRAMES_INPUT * 3  # нам ведь не нужны все данные, тут нам и меньше пойдет?

    score1 = -1  # признак, что модели нет в памяти
    if model != None and len(mfcc_buff) > FOREST_FRAMES_INPUT:
        # делаем предсказание аномальности
        line = get_line(mfcc_buff.items)
        y2 = model.predict(line)
        # y3 = model_n.predict(line)
        # signal_str = f'Forest predict {(1-y2[0])/2};  NormModel predict {y3[0]:.3};  stdscore {signal_str}'
        signal_str = f'Forest predict {(1-y2[0])/2}; stdscore {signal_str}'
        if score > 1.1 or y2[0] < 0 :
            logger.info(signal_str)
        if y2[0] < 0:
            score1 = 1.1
        else:
            score1 = 0
        score = score+score1

    smoothed_score = 0.8 * smoothed_score + 0.2 * score
    if learning_signal_left_count > 0 and score1 < 0:
        if model == None:
            print(f'Learning mode: {np.mean(np.abs(scaled_data_chunk)):.3} left_count={learning_signal_left_count}')
        else:
            print('Buffer filling', end='\r')
    else:
        print(signal_str, end='\r')
        if score > 1.5 or smoothed_score>1.1:
            print('\n') # перевод строки
            logger.warning('Anomaly alert: ' + signal_str)
            if need_to_save_mp3 < 0:  # если аномалия еще не пишется
                from datetime import datetime

                now = datetime.now()  # Получаем текущую дату и время
                # Преобразуем в строку в заданном формате
                filename = f"anomaly_{now.strftime('%Y%m%d-%H%M%S')}_{smoothed_score:.3}.mp3"
                # ставим признак записи MP3, записываем, сколько фреймов ждать до начала записи
                # ставим число таким, что бы было несколько секунд до аномалии и несколько после
                if need_to_save_mp3==-1: need_to_save_mp3 =  history_buff.max_items // 2

    if need_to_save_mp3 > 0: need_to_save_mp3 -= 1 # это в случае, если недавно обнаружили аномалию
    if need_to_save_mp3 < -1: need_to_save_mp3 += 1 # это в случае, если только что записали mp3
    if need_to_save_mp3 == 0:
        need_to_save_mp3 = -history_buff.max_items  # признак того, что записали mp3
        os.makedirs(MP3PATH) if not os.path.exists(MP3PATH) else None
        full_path = os.path.join(MP3PATH, filename)
        logger.info(f'Export to {full_path}')
        export_to_MP3(history_buff.get_items(), RATE, full_path, kbps=128)
        model_save(mfcc_buff, full_path + '.mfcc')
        logger.debug(f'Recorded {full_path}')
        if smoothed_score>1: smoothed_score = 1.00001
