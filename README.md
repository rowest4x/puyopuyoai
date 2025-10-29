# puyopuyoai

並列Actorと優先度付き経験再生を用いた深層強化学習によるぷよぷよAIです。2025年2月に知能と情報（日本知能情報ファジィ学会誌）に掲載された[論文](https://doi.org/10.3156/jsoft.37.1_501)のものです。ぷよぷよのロジックに自作ライブラリの[Puyothon](https://github.com/rowest4x/puyothon.git)を使っています。

研究用途にのみお使いください。

### 導入方法

1. 以下のコマンドでクローン
    ```bash
    git clone https://github.com/rowest4x/puyopuyoai.git
    ```
1. クローンしたディレクトリに移動
    ```bash
    cd puyopuyoai
    ```
1. 必要なら仮想環境を作成して起動
    ```bash
    python -m venv env
    source env/bin/activate
    ```

1. ライブラリをインストール
    ```bash
    pip install -r requirements.txt
    ```
    ※tensorflowのバージョンに合わせたCUDAやCUDNNが必要です。  
    ※[Puyothon](https://github.com/rowest4x/puyothon.git) のインストールに gcc 等の C コンパイラが必要です。

### ファイル構成

- **learn.py**  
    ぷよぷよAIの学習を行うPythonプログラムです。以下のように実行します。  
    ```
    python learn.py
    ```
    学習後のモデルは`log/<datetime>/model/`に保存されます。また、`log`ファイルには学習の経過も記録されます。  

- **testplay.py**  
    `learn.py`で学習したモデルを使って指定回数ゲームをプレイします。以下のように実行します。使用するモデルは`testplay.py`の中に直書きしているので都度書き換えてください。  
    ```
    python testplay.py
    ```
    プレイ結果は`log/<datetime>/testplay/`に保存されます。  

- **play_log.py**  
    `testplay.py`で行われたプレイを確認するプログラムです。以下のように実行します。  
    ```
    python play_log.py
    ```
    その後表示されたURL（http://127.0.0.1:5000 など）にアクセスしてください。  

- **model.py**  
    モデルの定義を行うファイルです。  

- **Player.py**  
    ぷよぷよのプレイをサポートするPlayerクラスを定義するファイルです。  

- **Actor.py**  
    学習のためにぷよぷよのプレイを行うActorのクラスを定義するファイルです。Playerクラスを継承しています。  

- **TsumoLoader.py**  
    ぷよぷよの配色パターンを読み込むTsumoLoaderクラスを定義するファイルです。  

- **tsumo/color_mappting.txt, tsumo/tsumo.txt**  
    あらかじめ生成されたぷよぷよの配色パターンです。『ぷよぷよeスポーツ』と同等のもの（と考えられるもの）を用いています。  

- **templetes/board.html, templetes/index.html**  
    play_log.pyで使われるhtmlファイルです。  

- **static/block.png など**  
    play_log.pyで描画に使用する画像ファイルです。  
