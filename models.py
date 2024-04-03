import numpy as np
import optuna
from loguru import logger
from scipy.stats import anderson
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from audio_anomaly import get_line, FOREST_FRAMES_INPUT, FLOAT_TYPE

Audio_norms_get_subsumms = 'None'

class AudioNormsModel:
    """
    Создание класса для отслеживания аномалий по величине множественной дисперсии.
    0. Собираются нормальные данные и в виде примитивов data поступают на вход.
    1. Примитивы нормируются.
    2. Используется генерация фич методом сумирования простейших примитивов.
    3. Получившиеся суммы снова нормируются и их общая сумма служит мерилом аномальности.
    """

    def __init__(self, codename, num_components=300, subsumm_len=30):
        self.num_components = num_components
        self.subsumm_len = subsumm_len
        self.scaler = StandardScaler()  # первый нормализатор. ДО подсуммирования
        self.codename = codename
        self.code = None  # кодированный словарь для составления сумм ()
        self.scaler_reduced = StandardScaler()  # второй нормализатор, ПОСЛЕ суммирования
        self.anomaly_max_percentile = 95 # какой прецентиль участвует в определении аномальности

    def init_codename(self):
        if os.path.isfile(f'./model/{self.codename}.py'):
            import importlib
            global Audio_norms_get_subsumms
            Audio_norms_get_subsumms = importlib.import_module(f'model.{self.codename}')
            return True
        else:
            raise Exception(f'File not found ./model/{self.codename}.py')

    def objective(self, trial, data, subsumm_len):
        subsumm = -1
        for i in range(subsumm_len):
            # выберем колонки и какой знак у колонок
            znak = trial.suggest_int(f'znak{i}', -1, 1)
            kolonka_nr = trial.suggest_int(f'col{i}', 0, data.shape[1] - 1)
            if znak < 0:
                kolonka = data[:, kolonka_nr].astype(np.float64)
            else:
                kolonka = data[:, kolonka_nr].astype(np.float64)

            if i == 0:
                subsumm = kolonka
                if znak < 0: subsumm = -kolonka
            else:
                if znak < 0:
                    subsumm -= kolonka
                else:
                    subsumm += kolonka
            # print(f'np.std(subsumm)={np.std(subsumm)}')
        # return np.std(subsumm) # требование минимальности стандартного отклонения
        # альтернатива - требование нормальности
        norm_test = anderson(subsumm)
        return norm_test.statistic / norm_test.critical_values[0]

    def learn_subsumm(self, data, optuna_n_trials=30):
        study = optuna.create_study()
        study.optimize(lambda trial: self.objective(trial, data, self.subsumm_len), n_trials=optuna_n_trials)
        return study.best_params

    def build_code(self, data, codes):
        str = 'import numba\n'
        str += 'import numpy as np\n'
        str += '#@numba.jit( )\n'
        str += 'def run(data):\n'
        str += f'\tresult=np.empty( ({len(codes)}, data.shape[1]), dtype=np.{data.dtype.name})\n'
        for i, code in enumerate(codes):
            row_code = f'\tresult[{i}] = '
            minuses=[]
            pluses=[]
            for i in range(0, self.subsumm_len):
                if code[f'znak{i}'] < 0:
                    minuses.append(code[f"col{i}"])
                else:
                    pluses.append(code[f"col{i}"])

            # for i in range(0, self.subsumm_len):
            #     if code[f'znak{i}'] < 0:
            #         row_code += '-'
            #     else:
            #         row_code += '+'
            #     row_code += f'data[{code[f"col{i}"]}]'
            # str += f'{row_code}\n'

            if len(pluses)>0:
                pluses.sort()
                row_code += f'+ np.sum(data[{pluses},], axis=0) '
            if len(minuses) > 0:
                minuses.sort()
                row_code += f'- np.sum(data[{minuses},], axis=0)'
            str += row_code+'\n'
        str += f'\treturn result\n'
        return str

    def learn_model(self, raw_buffer):
        if not isinstance(raw_buffer, list):
            raise TypeError("AudioNormsModel: Only list are allowed")
        test = get_line(raw_buffer, 0)
        X_train = np.empty((len(raw_buffer) - FOREST_FRAMES_INPUT, test.shape[1]), dtype=FLOAT_TYPE)
        logger.info(f'AudioNormsModel: Размерность обучающего набора {X_train.shape}')
        # X_train=X_train.reshape(1, X_train.shape[0])
        for get_subsumms in trange(0, len(raw_buffer) - FOREST_FRAMES_INPUT):
            X_train[get_subsumms] = get_line(raw_buffer, get_subsumms)[0]

        logger.info('Learning; Normalize data')
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)

        # после создания обучающего набора будем подбирать подсуммы
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info('Start optuna subsumms')
        code = []
        for get_subsumms in trange(0, self.num_components):
            code.append(self.learn_subsumm(X_train))
        self.code = code

        # теперь нужно сформировать сокращенную обучающую выборку
        logger.info('Start code generator')
        code_str = self.build_code(X_train, code)
        with open(f'./model/{self.codename}.py', "w") as text_file:
            text_file.write(code_str)

        self.init_codename()

        reduced = Audio_norms_get_subsumms.run(X_train).transpose()
        self.scaler_reduced.fit(reduced)
        final_data = self.scaler_reduced.transform(reduced)
        # scores = np.mean(np.abs(final_data), axis=1)
        scores = np.percentile(np.abs(final_data), self.anomaly_max_percentile, axis=1)
        self.treshold = np.percentile(scores, 99.95)
        logger.info(f'AudioNormsModel treshold= {self.treshold:.3}' )

    def predict(self, line):
        data_norm = self.scaler.transform(line)
        reduced = Audio_norms_get_subsumms.run(data_norm.transpose()).transpose()
        final_data = self.scaler_reduced.transform(reduced)
        # score = np.mean(np.abs(final_data))
        score = np.percentile(np.abs(final_data), self.anomaly_max_percentile, axis=1)
        return score / self.treshold
