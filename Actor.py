import puyothon as puyo
import numpy as np
from Player import Player

class Actor(Player):
    def __init__(self, board:np.ndarray=None, tsumo:np.ndarray=None, model=None):
        super().__init__(board=board, tsumo=tsumo)
        self.model = model
        self.step = 0

        self.Q_list = []
        self.r_list = []
        self.x_list = []
        self.dead_count = -1
        self.int_r = 0
        self.max_chain = 0
        
        self.action_log = []
    

    def getBoardForModel(self) -> np.ndarray:
        return puyo.cvtBoardForModel(self.board)
    

    def getAbleBoardsForModel(self) -> tuple[np.ndarray, np.ndarray]:
        p_puyo = self.tsumo[self.tsumo_count%self.tsumo_size]
        c_puyo = self.tsumo[(self.tsumo_count+1)%self.tsumo_size]
        boards, actions = puyo.getAbleBoardsForModel(self.board, p_puyo, c_puyo)
        return boards, actions
    

    def getNextPuyosForModel(self) -> list[int]:
        return self.tsumo[self.tsumo_count : self.tsumo_count+4].tolist()
    

    def copy(self) -> 'Actor':
        c_actor = Actor(board=self.board.copy(), tsumo=self.tsumo, model=self.model)
        c_actor.tsumo_count = self.tsumo_count
        c_actor.score = self.score
        c_actor.chain_count = self.chain_count
        return c_actor
    

    def putPuyoByNum(self, action:int) -> bool:
        a = action
        if a >= 3: a += 1
        if a >= 21: a += 1

        col = a // 4 + 1
        rot = a % 4

        if self.putPuyo(action=(col, rot)):
            self.action_log.append(action)
            return True
        else:
            return False
    

    def putPuyoForLearning(self, action_list:list[int], Q_list:np.ndarray, chosen_action:int=-1, discount:float=0.98, multi_step_num:int=3):
        Q_chosen = 0 # 実際に選択されるactionのQ値
        maxQ = 0 # 一番大きいQ値
        r = 0 # 報酬
        x = None # 選ばれた場所にぷよが設置された後の盤面
        chain_count = 0

        if self.dead_count == -1:
            i_maxQ = np.argmax(Q_list)
            maxQ = Q_list[i_maxQ]
            action_maxQ = action_list[i_maxQ] # actionとQは一対一対応しているのでこうできる
            if chosen_action == -1: # chosen_actionが指定されていないとき
                chosen_action = action_maxQ
            Q_chosen = Q_list[action_list.index(chosen_action)]

            # ぷよの設置と連鎖の実行
            self.putPuyoByNum(chosen_action)
            x = self.getBoardForModel()
            chain_count, score = self.chainAuto()
            r = score
        
        # ゲームオーバーしたとき
        if self.isDead():
            self.dead_count += 1
        if self.dead_count == 0:
            r = -1000 # ゲームオーバーは減点
        elif self.dead_count > 0:
            r = 0
        
        # Multi-step Learning
        td_n_ago = None
        x_n_ago = None
        Q_n_ago = None
        if(self.step >= multi_step_num):
            td_n_ago = discount**multi_step_num * maxQ - self.Q_list[-multi_step_num]
            td_n_ago += sum([discount**(multi_step_num - i) * self.r_list[-i] for i in range(1, multi_step_num+1)])
            x_n_ago = self.x_list[-multi_step_num]
            Q_n_ago = self.Q_list[-multi_step_num]

        # 記録
        self.x_list.append(x)
        self.Q_list.append(Q_chosen)
        self.r_list.append(r)

        self.int_r += r
        if chain_count > self.max_chain:
            self.max_chain = chain_count

        self.step += 1
        return x_n_ago, Q_n_ago, td_n_ago
    
    
    def getBoardForSearch1(self, action1:np.ndarray, y1:np.ndarray, n=5):
        i1 = np.argsort(y1)[::-1][:n]
        action_rank1 = action1[i1]
        q1 = y1[i1]

        actor1 = []
        r1 = []
        for i in range(i1.shape[0]):
            actor = self.copy()
            actor.putPuyoForLearning([action_rank1[i]], np.array([q1[i]]))
            if not actor.isDead():
                actor1.append(actor)
                r = actor.int_r - self.int_r
                r1.append(r)
        r1 = np.array(r1)
        
        x2 = []
        action2 = []
        owner2 = []
        board_num = 0
        for i, actor in enumerate(actor1):
            boards, actions = actor.getAbleBoardsForModel()
            x2.append(boards)
            action2.append(actions)
            owner2.append(np.full(actions.shape, i))
            board_num += boards.shape[0]
        
        return board_num, x2, (action_rank1, r1, actor1, action2, owner2)
    

    def getBoardForSearch2(self, y2:np.ndarray, tmp, n=10):
        action_rank1, r1, actor1, action2, owner2 = tmp
        if y2.shape[0] == 0:
            return 0, [], ([], [], [], [], [])
        action2 = np.hstack(action2)
        owner2 = np.hstack(owner2)
            
        i2 = np.argsort(y2+r1[owner2])[::-1][:n]
        action_rank2 = action2[i2]
        actor_rank2 = owner2[i2]
        q2 = y2[i2]

        actor2 = []
        r2 = []
        for i in range(i2.shape[0]):
            actor = actor1[actor_rank2[i]].copy()
            actor.putPuyoForLearning([action_rank2[i]], np.array([q2[i]]))
            if not actor.isDead():
                actor2.append(actor)
                r = actor.int_r - self.int_r
                r2.append(r)
        r2 = np.array(r2)

        x3 = []
        action3 = []
        owner3 = []
        board_num = 0
        for i, actor in enumerate(actor2):
            boards, actions = actor.getAbleBoardsForModel()
            x3.append(boards)
            action3.append(actions)
            owner3.append(np.full(actions.shape, i))
            board_num += boards.shape[0]
        return board_num, x3, (r2, action_rank1, actor_rank2, action3, owner3)
    

    def putPuyoForSearch3(self, y3:np.ndarray, tmp) -> None:
        r2, action_rank1, actor_rank2, action3, owner3 = tmp
        if y3.shape[0] == 0:
            return
        action3 = np.hstack(action3)
        owner3 = np.hstack(owner3)
        
        i_max = np.argmax(y3+r2[owner3])
        action_max = action3[i_max]
        actor_max = owner3[i_max]

        chosen_action = action_rank1[actor_rank2[actor_max]]
        self.putPuyoForLearning([chosen_action], [0])
    

    def putPuyoBySearch(self) -> None:
        x1, action1 = self.getAbleBoardsForModel()
        y1 = self.model.predict(x1, verbose=0).reshape(x1.shape[0])
        
        i1 = np.argsort(y1)[::-1][:10]
        action_rank1 = action1[i1]
        q1 = y1[i1]

        actor1 = []
        r1 = []
        for i in range(i1.shape[0]):
            actor = self.copy()
            actor.putPuyoForLearning([action_rank1[i]], np.array([q1[i]]))
            actor1.append(actor)
            r = actor.int_r - self.int_r
            r1.append(r)
        r1 = np.array(r1)

        x2 = []
        action2 = []
        owner2 = []
        for i, actor in enumerate(actor1):
            boards, actions = actor.getAbleBoardsForModel()
            x2.append(boards)
            action2.append(actions)
            owner2.append(np.full(actions.shape, i))
        x2 = np.vstack(x2)
        y2 = self.model.predict(x2, verbose=0).reshape(x2.shape[0])
        action2 = np.hstack(action2)
        owner2 = np.hstack(owner2)
        i2 = np.argsort(y2+r1[owner2])[::-1][:20]
        action_rank2 = action2[i2]
        actor_rank2 = owner2[i2]
        q2 = y2[i2]

        actor2 = []
        r2 = []
        for i in range(i2.shape[0]):
            actor = actor1[actor_rank2[i]].copy()
            actor.putPuyoForLearning([action_rank2[i]], np.array([q2[i]]))
            actor2.append(actor)
            r = actor.int_r - self.int_r
            r2.append(r)
        r2 = np.array(r2)


        x3 = []
        action3 = []
        owner3 = []
        for i, actor in enumerate(actor2):
            boards, actions = actor.getAbleBoardsForModel()
            x3.append(boards)
            action3.append(actions)
            owner3.append(np.full(actions.shape, i))
        x3 = np.vstack(x3)
        y3 = self.model.predict(x3, verbose=0).reshape(x3.shape[0])
        action3 = np.hstack(action3)
        owner3 = np.hstack(owner3)

        i_max = np.argmax(y3+r2[owner3])
        action_max = action3[i_max]
        actor_max = owner3[i_max]

        chosen_action = action_rank1[actor_rank2[actor_max]]
        self.putPuyoByNum(chosen_action)


    def putPuyoForModel(self, action:int, q:np.ndarray) -> int:
        if(self.putPuyoByNum(action)):
            return action
        
        for _action in np.argsort(q)[::-1]:
            if(self.putPuyoByNum(_action)):
                return _action


        return action
