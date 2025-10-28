import numpy as np
import random
import puyothon as puyo
from Player import Player
from Actor import Actor
from TsumoLoader import TsumoLoader
from deep import makeModel, Res_Block
import time
import tensorflow as tf
from itertools import permutations
import os
import datetime
import json
import io


# Mixed Precisionのポリシーを設定
# ぷよぷよは連鎖が伸びるとスコアが爆発的に増加するので、16bitだとうまくいかなかった
# なのでコメントアウトしている
#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#policy = tf.keras.mixed_precision.Policy('float32')
#tf.keras.mixed_precision.set_global_policy(policy)


# 乱数seed関連の設定============================================================================================
# tensorflow 2.09以前では再現性が確保できていたが、2.10以降では未確認。消してもいいかも

# 環境変数の設定
os.environ['PYTHONHASHSEED'] = '100'

# Pythonの乱数シードの設定
random.seed(100)

# NumPyの乱数シードの設定
np.random.seed(100)

# TensorFlowの乱数シードの設定
tf.random.set_seed(100)
tf.keras.utils.set_random_seed(100)

# TensorFlowでの再現性を確保するための設定
# デバイスの設定
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 再現性のために非決定性の操作を避ける
# 詳細: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
tf.config.experimental.enable_op_determinism()

# 必要に応じて以下のような設定も追加できます
# tf.compat.v1.set_random_seed(100)


# ハイパーパラメタ ================================================================================================
GAME_NUM = 1000 # 並列Actorの数
MOVE_PER_GAME = 50 # 何手でゲームを終了するか
GENERATION_NUM = 100 # 学習の世代数

EPSILON = 0.4 # Ape-Xのε
ALPHA = 7 # Ape-Xのα
epsilons = np.power(EPSILON , 1 + ALPHA * np.arange(GAME_NUM) / (GAME_NUM - 1)) # Ape-Xのε_i

DISCOUNT = 0.995 # 割引率

MULTI_STEP_NUM = 3 # multi-step learning のステップ数

TRAIN_SIZE = 100000 # 一世代あたりに学習する訓練データの数
BATCH_SIZE = 256


#  GAME_NUM個のゲームを並列にプレイする関数 ===========================================================================================
def play(model):
    # 学習データ用バッファ
    x_train = []
    Q_train = []
    td_train = []

    # 世代終了時に取得するログ用
    maxchains : np.ndarray
    intrewards : np.ndarray

    # 並列Actorの生成
    tsumo_loader = TsumoLoader()
    actors : list[Actor] = []
    for i in range(GAME_NUM):
        seed : int  = random.randrange(0xffff)
        tsumo = tsumo_loader.load(seed)
        actors.append(Actor(tsumo=tsumo))
    
    # ゲーム終了まで並列プレイ
    for step in range(MOVE_PER_GAME + MULTI_STEP_NUM):

        # 最後のターンが終了したら最大連鎖数と累積報酬を記録する
        if step == MOVE_PER_GAME:
            maxchains = np.array([actor.max_chain for actor in actors])
            intrewards = np.array([actor.int_r for actor in actors])


        # 各Actorからboardsとactionsを取得
        # 盤面と合法手は1対1対応している
        boards = [] # モデルに渡す盤面のリスト
        actions = [] # 各Actorごとの合法手のリスト
        alive_actors = []
        for i, actor in enumerate(actors):
            board, action = actor.getAbleBoardsForModel()
            action = action.tolist()
            if actor.dead_count == -1:
                alive_actors.append(i)
                boards.append(board)
            elif actor.dead_count < MULTI_STEP_NUM:
                alive_actors.append(i)
            actions.append(action)
        
        if not boards:
            continue

        # モデルでQ値を予測
        # 一度に全てのActorの予測を行うことでGPUの利用効率を高めている
        x = np.vstack(boards)
        y = model.predict(x, verbose=0)

        # ぷよの設置と学習用データの記録
        i = 0
        for game in alive_actors:
            i_next = i + len(actions[game])
            
            # ε-greedyによって選択する手を決める
            # 後の処理では、chosen_actionが-1のときはQ値が一番高い手を選び、chosen_actionが0以上のときはその手を選ぶ
            chosen_action = -1
            if actors[game].dead_count == -1 and random.random() <= epsilons[game]:
                chosen_action = random.choice(actions[game])
            
            # ぷよを設置して学習用の盤面、Q値、td誤差を取得する
            # MULTI_STEP_NUMだけ遅れて学習データが返ってくる（Q値の計算のためにMULTI_STEP_NUM分先の報酬を見なければならないので）
            pre_x, pre_Q, pre_td = actors[game].putPuyoForLearning(actions[game], y[i:i_next], chosen_action=chosen_action, discount=DISCOUNT, multi_step_num=MULTI_STEP_NUM)
            
            # MULTI_STEP_NUMだけ遅れているため、最初のMULTI_STEP_NUM回は記録しない
            if step >= MULTI_STEP_NUM:
                x_train.append(pre_x)
                Q_train.append(pre_Q)
                td_train.append(pre_td)

            i = i_next
    
    return np.vstack(x_train), np.array(Q_train), np.array(td_train), maxchains, intrewards



# モデルの作成・読み込み =================================================================================================================
start_time_all = time.time()
time_bias = 0
model_num = -1

# 新しく学習するとき
model = makeModel()
datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
log_dir_path = f"log/{datetime_str}"
os.makedirs(log_dir_path)
os.mkdir(log_dir_path + "/model")
os.mkdir(log_dir_path + "/maxchain_hist_log")
os.mkdir(log_dir_path + "/intreward_log")
# 新しく学習するときここまで


# 途中から学習するとき（モデルのパスを指定する）
# log_dir_path = "log/20250322172134"
# model_num = 2
# model = tf.keras.models.load_model(log_dir_path + f"/model/{model_num:0>4}.keras", custom_objects={'Res_Block': Res_Block})
# with open(log_dir_path + "/lean_log.ndjson", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     if lines:
#         last_line = lines[model_num]
#         last_log = json.loads(last_line)
#         time_bias = last_log["time_all"]
# 途中から学習するときここまで



# 学習 ===============================================================================================================================
hist_log = np.zeros((GENERATION_NUM, 20), dtype=np.int32)
reward_log = np.zeros((GENERATION_NUM, GAME_NUM))
for generation in range(model_num + 1, GENERATION_NUM):

    start_time = time.time()
    
    #1000ゲーム並列プレイ
    x_train, Q_train, td_train, maxchains, intrewards = play(model)
    
    #平均最大連鎖数と平均累積報酬を計算，連鎖数をヒストグラムに
    mean_maxchain = np.mean(maxchains)
    mean_intreward = np.mean(intrewards)
    hist, bins = np.histogram(maxchains, bins=[i-0.5 for i in range(21)])
    hist_log[generation, :] = hist
    reward_log[generation, :] = intrewards
    hist = [f"{n:>3}" for n in hist]
    
    # 学習するためにtd誤差を計算（色を入れ替えた物についても計算）
    N = x_train.shape[0]
    COLOR_NUM = 4
    PERMUTATION_NUM = 4 * 3 * 2 * 1
    x_train = np.vstack([x_train[:, :, :, c] for c in permutations(range(COLOR_NUM))])
    Q_train = Q_train.reshape(1, N).repeat(PERMUTATION_NUM, axis=0).reshape(N*PERMUTATION_NUM)
    td_train = td_train.reshape(1, N).repeat(PERMUTATION_NUM, axis=0).reshape(N*PERMUTATION_NUM)

    #優先度付き経験再生
    alpha = 0.5
    priorities = np.power(np.abs(td_train) + 1e-15, alpha)
    probs = priorities / priorities.sum()

    # 優先度付き経験再生の重みづけ
    # だいぶ前に試したときは重みづけしない方がうまくいったので消したままにしている（もしかしたら学習回数が足りてないだけかも）
    # beta = 0.4 + generation*0.6/(GENERATION_NUM-1)
    # weights = 1 / (N*24 * probs) ** beta
    # weights = weights / np.max(weights)
    # y_train = weights * td_train + Q_train

    # 学習データ
    y_train = td_train + Q_train

    # 学習
    indices = np.random.choice(np.arange(N*PERMUTATION_NUM), p=probs, replace=False, size=TRAIN_SIZE) #優先度の上位から取ってくる
    model.fit(x_train[indices], y_train[indices], batch_size=BATCH_SIZE, epochs=1, verbose=1)

    # 進捗を出力
    now_time = time.time()
    print(f"time:{now_time-start_time:.2f}s, generation:{generation:>3}, mean_intreward:{mean_intreward:.3f}, mean_maxchain:{mean_maxchain:.3f}, max_chain hist:{hist}")

    # 保存
    model.save(log_dir_path + f"/model/{generation:0>4}.keras")
    np.save(log_dir_path + f"/maxchain_hist_log/{generation:0>4}.npy", hist_log)
    np.save(log_dir_path + f"/intreward_log/{generation:0>4}.npy", reward_log)
    with open(log_dir_path + "/learn_log.ndjson", "a", encoding="utf-8") as f:
        data = {"time_all" : now_time-start_time_all+time_bias, "time":now_time-start_time, "generation":generation, "mean_intreward":mean_intreward, "mean_maxchain":mean_maxchain, "max_chain_hist":hist}
        f.write(json.dumps(data, ensure_ascii=False) + "\n")



# 学習にかかった時間を出力
end_time = time.time()
total_time = int(end_time - start_time_all + time_bias)
print(f"total_time:{total_time//3600}h{(total_time//60)%60}m{total_time%60}s ({total_time}s)")

# 学習時の条件などを記録
with open(log_dir_path + "/conditions.json", "w", encoding="utf-8") as f:
    # model.summary()を文字列として取得
    summary_io = io.StringIO()
    model.summary(print_fn=lambda x: summary_io.write(x + "\n"))
    model_summary_str = summary_io.getvalue()

    data = {"GAME_NUM" : GAME_NUM,
            "MOVE_PER_GAME" : MOVE_PER_GAME,
            "GENERATION_NUM" : GENERATION_NUM,
            "EPSILON" : EPSILON,
            "ALPHA" : ALPHA,
            "DISCOUNT" : DISCOUNT,
            "model": model_summary_str,
            "total_time" : f"{total_time//3600}:{(total_time//60)%60:0>2}:{total_time%60:0>2}"}

    json.dump(data, f, ensure_ascii=False, indent=4)