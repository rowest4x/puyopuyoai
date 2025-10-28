from flask import Flask, render_template, request, jsonify
import puyothon as puyo
from Player import Player
from TsumoLoader import TsumoLoader
import numpy as np
import os

def getBoardTsumo(player:Player):
    result = np.zeros((15, 9))
    result[:, :-1] = player.getBoardForDraw().T

    hand, next, nxnx = player.getNextPuyosForDraw()
    result[1][3] = hand[0]
    result[0][3] = hand[1]
    result[1][8] = next[0]
    result[0][8] = next[1]
    result[4][8] = nxnx[0]
    result[3][8] = nxnx[1]
    return result

def genGamedata(tsumo_seed, action_log):
    print(f"tsumo_seed:{tsumo_seed}")
    tsumo_loader = TsumoLoader()
    tsumo = tsumo_loader.load(tsumo_seed)
    color_map = tsumo_loader.load_mapping(tsumo_seed)

    # ぷよのマッピング（画像のパスを指定）
    image_map = {
        -2: 'static/ojama.png', #おじゃまぷよ
        -1: 'static/block.png', #壁用ブロック
        0: 'static/empty.png',  # 空マス
    }
    for key in color_map:
        img_path = {
            "r": 'static/red.png',    # 赤ぷよ
            "g": 'static/green.png',  # 緑ぷよ
            "b": 'static/blue.png',   # 青ぷよ
            "y": 'static/yellow.png', # 黄ぷよ
            "p": 'static/purple.png'  # 紫ぷよ
        }[color_map[key]]
        image_map[key] = img_path

    # 盤面の再現
    player = Player(tsumo=tsumo)
    boards = [getBoardTsumo(player)]
    scores = [0]
    chain_counts = [0]
    all_clears = [False]

    for a in action_log:
        if a < 0:
            break

        if a >= 3: a += 1
        if a >= 21: a += 1
        col = a // 4 + 1
        rot = a % 4
        player.putPuyo((col, rot))
        
        boards.append(getBoardTsumo(player))
        scores.append(player.score)
        chain_counts.append(player.chain_count)
        all_clears.append(player.isAllClear())

        while player.next():
            boards.append(getBoardTsumo(player))
            scores.append(player.score)
            chain_counts.append(player.chain_count)
            all_clears.append(player.isAllClear())

    return boards, scores, chain_counts, all_clears, image_map


app = Flask(__name__)

tsumo_seeds : np.ndarray
action_logs : np.ndarray

game_id :int
boards: list
scores : list
chain_counts : list
all_clears : list

def list_testplay_dirs():
    log_dirs = os.listdir("log")
    testplay_dirs = []
    for log_dir in log_dirs:
        if os.path.isdir(f"log/{log_dir}/testplay"):
            testplay_dirs.extend([f"log/{log_dir}/testplay/" + d for d in os.listdir(f"log/{log_dir}/testplay")])
    return sorted(testplay_dirs)

@app.route('/', methods=['GET', 'POST'])
def index():
    global boards, scores, chain_counts, all_clears, image_map, game_id
    global tsumo_seeds, action_logs

    dirs = list_testplay_dirs()

    if request.method == 'POST':
        # ゲーム番号選択フォームの場合は "game_id" が含まれている
        if 'game_id' in request.form:
            game_id = int(request.form['game_id'])
            # hidden で渡された log_dir を取得
            log_dir = request.form.get('log_dir', '')
            boards, scores, chain_counts, all_clears, image_map = genGamedata(tsumo_seeds[game_id], action_logs[game_id])
            return render_template('index.html',
                                   dirs=dirs,
                                   log_dir=log_dir,
                                   game_num=len(action_logs),
                                   game_id=game_id,
                                   max_turn=len(boards)-1,
                                   turn=0)

        # ファイル選択フォームの場合
        elif 'log_dir' in request.form:
            log_dir = request.form['log_dir']
            tsumo_seeds = np.load(f"{log_dir}/tsumo_seeds.npy")
            action_logs = np.load(f"{log_dir}/action_logs.npy")
            game_id = 0
            boards, scores, chain_counts, all_clears, image_map = genGamedata(tsumo_seeds[game_id], action_logs[game_id])
            return render_template('index.html',
                                   dirs=dirs,
                                   log_dir=log_dir,
                                   game_num=len(action_logs),
                                   game_id=game_id,
                                   max_turn=len(boards)-1,
                                   turn=0)

    return render_template('index.html',
                           dirs=dirs,
                           log_dir='',
                           game_num=0,
                           game_id=0,
                           max_turn=0,
                           turn=0)

@app.route("/get_board/<int:turn>")
def get_board(turn):
    if 0 <= turn < len(boards):
        board_html = render_template("board.html", board=boards[turn], score=f"{scores[turn]:0>10}", chain_count=f"{chain_counts[turn]} chains!" if chain_counts[turn] > 0 else "", all_clear="All Clear" if all_clears[turn] else "", image_map=image_map)
        return jsonify({"html": board_html})
    return jsonify({"error": "Invalid turn"}), 400

if __name__ == '__main__':
    app.run(debug=True)