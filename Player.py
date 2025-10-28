import puyothon as puyo
import numpy as np

class Player:
    def __init__(self, board:np.ndarray=None, tsumo:np.ndarray=None):
        self.board : np.ndarray
        if(type(board) == type(None)):
            self.board = puyo.makeBoard()
        else:
            self.board = board
        
        self.tsumo : np.ndarray
        if(type(tsumo) != type(None)):
            self.tsumo = tsumo
        self.tsumo_size = tsumo.shape[0]

        self.tsumo_count : int = 0

        self.hand_puyo_col : int = 3
        self.hand_puyo_rot : int = 0

        self.score : int = 0
        self.chain_count : int = 0
    
    def setTsumo(self, tsumo:np.ndarray) -> 'Player':
        self.tsumo = tsumo
        return self
    
    def getBoardForDraw(self) -> np.ndarray:
        return self.board[puyo.PUYO, ::-1, :].T

    def getHandPuyo(self) -> tuple[int, int]:
        return (self.tsumo[self.tsumo_count], self.tsumo[self.tsumo_count+1])
    
    def getNextPuyosForDraw(self) -> tuple[tuple[int, int], tuple[int,int], tuple[int, int]]:
        hand = (self.tsumo[(self.tsumo_count)%self.tsumo_size], self.tsumo[(self.tsumo_count+1)%self.tsumo_size])
        next = (self.tsumo[(self.tsumo_count+2)%self.tsumo_size], self.tsumo[(self.tsumo_count+3)%self.tsumo_size])
        nxnx = (self.tsumo[(self.tsumo_count+4)%self.tsumo_size], self.tsumo[(self.tsumo_count+5)%self.tsumo_size])
        if(self.onChain()):
            return ((puyo.EMPTY, puyo.EMPTY), hand, next)
        else:
            return (hand, next, nxnx)
        
    def getAction(self) -> tuple[int, int]:
        rot = ((self.hand_puyo_rot % 4) + 4) % 4
        return (self.hand_puyo_col, rot)
    
    # 連鎖中ならTrue、そうでないならFalseを返す
    def onChain(self) -> bool:
        return self.copy().next()
    
    # 全消しが発生したときだけTrueを返す
    def isAllClear(self) -> bool:
        if self.score <= 0:
            return False
        
        has_color_puyo = (self.board > 0).any()
        has_ojama_puyo = (self.board == puyo.OJAMA).any()
        has_puyo = has_color_puyo or has_ojama_puyo
        return not has_puyo

    def isDead(self):
        return puyo.isDead(self.board)

    def copy(self) -> 'Player':
        c_player = Player(board=self.board.copy(), tsumo=self.tsumo)
        c_player.tsumo_count = self.tsumo_count
        c_player.score = self.score
        c_player.chain_count = self.chain_count
        return c_player
    
    # 連鎖を1段階進める関数（描画用）
    # 連鎖の処理が発生した場合はTrue、そうでない場合はFalseを返す
    def next(self) -> bool:
        #ぷよを消す処理
        score = puyo.erasePuyo(self.board, self.chain_count+1)
        if(score > 0):
            self.score += score
            self.chain_count += 1
            return True
        
        #ぷよを落とす処理
        fall_max = puyo.fallPuyo(self.board)
        if(fall_max > 0):
            return True
        
        #全消し判定
        if self.isAllClear():
            self.score += 2100
        
        if self.chain_count > 0:
            self.chain_count = 0
            return True
        
        return False
    
    # 連鎖を最後まで進める関数（学習用）
    def chainAuto(self) -> tuple[int, int]:
        chain_num, score = puyo.chainAuto(self.board)
        if self.isAllClear():
            score += 2100
        self.score += score
        return chain_num, score

    def putPuyo(self, action:tuple[int, int]=None) -> bool:
        puyo1, puyo2 = self.getHandPuyo()
        
        col:int
        rot:int
        if type(action) == type(None):
            col = self.hand_puyo_col
            rot = self.hand_puyo_rot
        else:
            col, rot = action

        if puyo.putPuyo(self.board, puyo1, puyo2, col, rot):
            self.tsumo_count = (self.tsumo_count + 2) % self.tsumo_size
            self.hand_puyo_col = 3
            self.hand_puyo_rot = 0
            return True
        else:
            return False
    
    # 盤面をリセット
    def reset(self) -> None:
        self.board = puyo.makeBoard()
        self.tsumo_count : int = 0

        self.hand_puyo_col : int = 3
        self.hand_puyo_rot : int = 0

        self.score : int = 0
        self.chain_count : int = 0

    # 盤面をリセットし、ツモも変更
    def new(self, tsumo:np.ndarray) -> None:
        self.board = puyo.makeBoard()

        self.tsumo = tsumo
        self.tsumo_count : int = 0

        self.hand_puyo_col : int = 3
        self.hand_puyo_rot : int = 0

        self.score : int = 0
        self.chain_count : int = 0
    
    # 手動操作用関数 ====================================
    def rotR(self) -> None:
        self.hand_puyo_rot += 1
        self.collision()
    
    def rotL(self) -> None:
        self.hand_puyo_rot -= 1
        self.collision()
    
    def moveR(self) -> None:
        self.hand_puyo_col += 1
        self.collision()
    
    def moveL(self) -> None:
        self.hand_puyo_col -= 1
        self.collision()
    
    def collision(self) -> None:
        col, rot = self.getAction()
        if(col < 1):
            self.hand_puyo_col = 1
        elif(col > puyo.COLS_NUM-2):
            self.hand_puyo_col = puyo.COLS_NUM - 2
        
        col, rot = self.getAction()
        if(rot == 1 and col+1 == puyo.COLS_NUM-1):
            self.hand_puyo_col -= 1
        elif(rot == 3 and col-1 == 0):
            self.hand_puyo_col += 1