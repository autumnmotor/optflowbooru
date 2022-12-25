# オプティカルフローによって画像の動きに注目するdeepbooruのスクリプト(WebUI's batch img2img)

オプティカルフローによって前後の画像から動きを検出し、動いた部分を重点的にdeepbooruし、自動的にpromptを生成する。

画像を用意したいな。

パラメータ

deepbooru limits per one image:　1つの画像に対して行うdeepbooruの上限。デフォルト:10
Mute word(Regex): deepbooruによって得られたワード群から除外するワード（正規表現で記述）
OptFlowbooru Token limits: 1つの画像に対して、このスクリプトによって得られたワードトークン数の制限（手動で設定したプロンプトのトークン数はカウントしない）。

描き途中。
