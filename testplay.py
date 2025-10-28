import numpy as np
import random
import puyothon as puyo
from Actor import Actor
from deep import makeModel, Res_Block
import time
import tensorflow as tf
import datetime
import os
import json
from TsumoLoader import TsumoLoader

# Mixed Precisionのポリシーを設定
# ぷよぷよは連鎖が伸びるとスコアが爆発的に増加するので、16bitだと桁数が足りなくてうまくいかなかった（なのでコメントアウトしている）
#policy = tf.keras.mixed_precision.Policy('mixed_float16')
#policy = tf.keras.mixed_precision.Policy('float32')
#tf.keras.mixed_precision.set_global_policy(policy)

# パラメタ =======================================================================
GAME_NUM = 1000
MOVE_PER_GAME = 50

# ディレクトリとモデルの指定
LOG_DIR_PATH = "/mnt/c/Users/rowes/wsl/puyo/log/20251028024300"
MODEL_NUM = 4

# 関数定義 =======================================================================
def getBoardsActionsAlives(actors:list[Actor]):
    boards = []
    actions = []
    
    alive_actors = []
    for i, actor in enumerate(actors):
        board, action = actor.getAbleBoardsForModel()
        action = action.tolist()
        if actor.dead_count == -1:
            alive_actors.append(i)
            boards.append(board)
            actions.append(action)
    return boards, actions, alive_actors

# GAME_NUM個のゲームを並列にプレイする関数
def play(model):
    tsumo_loader = TsumoLoader()
    actors : list[Actor] = []
    tsumo_seeds = []
    for i in range(GAME_NUM):
        seed : int  = random.randrange(0xffff)
        tsumo_seeds.append(seed)
        tsumo = tsumo_loader.load(seed)
        actors.append(Actor(tsumo=tsumo, model=model))
    
    for step in range(MOVE_PER_GAME):
        boards1, actions1, alive_actors = getBoardsActionsAlives(actors)
        x1 = np.vstack(boards1)
        if x1.shape[0] == 0:
            break
        y1 = model.predict(x1, verbose=0).reshape(x1.shape[0])
        i = 0
        boards2 = []
        board_num2 = [0]
        tmp2 = []
        for alive_actor_i, actor_i in enumerate(alive_actors):
            i_next = i + len(actions1[alive_actor_i])
            board_num, board, tmp = actors[actor_i].getBoardForSearch1(np.array(actions1[alive_actor_i]), y1[i:i_next])
            boards2.extend(board)
            board_num2.append(board_num + board_num2[-1])
            tmp2.append(tmp)
            i = i_next
        
        x2 = np.vstack(boards2)
        if x2.shape[0] == 0:
            break
        y2 = model.predict(x2, verbose=0).reshape(x2.shape[0])
        boards3 = []
        board_num3 = [0]
        tmp3 = []
        for alive_actor_i, actor_i in enumerate(alive_actors):
            i = alive_actor_i
            board_num, board, tmp = actors[actor_i].getBoardForSearch2(y2[board_num2[i]:board_num2[i+1]], tmp2[alive_actor_i])
            boards3.extend(board)
            board_num3.append(board_num+board_num3[-1])
            tmp3.append(tmp)
        
        x3 = np.vstack(boards3)
        if x3.shape[0] == 0:
            break
        y3 = model.predict(x3, verbose=0).reshape(x3.shape[0])
        for alive_actor_i, actor_i in enumerate(alive_actors):
            i = alive_actor_i
            actors[actor_i].putPuyoForSearch3(y3[board_num3[i]:board_num3[i+1]], tmp3[alive_actor_i])
        
        print(f"\rProcessing... {step}/{MOVE_PER_GAME}", end="")
        
    maxchains = np.array([actor.max_chain for actor in actors])
    intscores = np.array([actor.score for actor in actors])
    action_logs = np.array([actor.action_log + [-1] * (MOVE_PER_GAME - len(actor.action_log)) for actor in actors])
    print("")
    return maxchains, intscores, tsumo_seeds, action_logs


# テストプレイ======================================================================================================

model_path = f"{LOG_DIR_PATH}/model/{MODEL_NUM:0>4}.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'Res_Block': Res_Block})

model.summary()
print("")
print(f"GAME_NUM:{GAME_NUM}")
print(f"MOVE_PER_GAME:{MOVE_PER_GAME}")
print(f"model:{model_path}")
print("")

# 結果保存用のディレクトリ作成
datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
result_dir_path = f"{LOG_DIR_PATH}/testplay/model{MODEL_NUM:0>4}_{datetime_str}"
os.makedirs(result_dir_path)


# プレイ開始
start_time = time.time()

maxchains, intscores, tsumo_seeds, action_logs = play(model)

mean_maxchain = np.mean(maxchains)
mean_intscore = np.mean(intscores)
hist, bins = np.histogram(maxchains, bins=[i-0.5 for i in range(21)])
hist = [f"{n:>3}" for n in hist]

end_time = time.time()
print(f"time:{end_time-start_time:.2f}s, mean_intreward:{mean_intscore:.3f}, mean_maxchain:{mean_maxchain:.3f}, max_chain hist:{hist}")
# プレイ終了


# 結果を保存
with open(result_dir_path + "/testplay.json", "w", encoding="utf-8") as f:
    data = {"GAME_NUM" : GAME_NUM,
            "MOVE_PER_GAME" : MOVE_PER_GAME,
            "model" : model_path,
            "time" : end_time-start_time,
            "mean_intscore" : mean_intscore,
            "mean_maxchain" : mean_maxchain}
    json.dump(data, f, ensure_ascii=False, indent=4)
    
np.save(result_dir_path + "/tsumo_seeds", tsumo_seeds)
np.save(result_dir_path + "/action_logs", action_logs)
