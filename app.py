"""Flask web server for Two-Sets-of-3 (6-card Family Rummy)."""

import threading

from flask import Flask, jsonify, render_template_string, request

from game import GameEngine, TrajectoryLogger

app = Flask(__name__)

LOG_FILE = 'trajectories.jsonl'
game_lock = threading.Lock()
trajectory_logger = TrajectoryLogger(LOG_FILE)
engine = GameEngine(logger=trajectory_logger)


def get_client_ip():
  # Allows ?mock_ip=x in URL when testing multiple tabs on one laptop
  return request.args.get('mock_ip') or request.remote_addr


def get_json_body():
  return request.get_json(silent=True) or {}


# --- WEB API ROUTES ---
@app.route('/')
def home():
  return render_template_string(HTML_PAGE)


@app.route('/join', methods=['POST'])
def join_game():
  ip = get_client_ip()
  body = get_json_body()
  raw_name = body.get('name', '')
  if not isinstance(raw_name, str):
    raw_name = ''
  with game_lock:
    result = engine.join_player(ip, name=raw_name, player_type='human')
  return jsonify(result)


@app.route('/start', methods=['POST'])
def start_route():
  with game_lock:
    ok = engine.start_match()
  return jsonify({'ok': ok})


@app.route('/rematch', methods=['POST'])
def rematch_route():
  with game_lock:
    ok = engine.rematch()
  return jsonify({'ok': ok})


@app.route('/restart', methods=['POST'])
def restart_route():
  with game_lock:
    engine.reset_lobby()
  return jsonify({'ok': True})


@app.route('/state')
def get_state():
  ip = get_client_ip()
  with game_lock:
    state = engine.get_ui_state(ip)
  return jsonify(state)


@app.route('/reorder', methods=['POST'])
def reorder_card():
  ip = get_client_ip()
  idx = get_json_body().get('index', -1)
  with game_lock:
    p = engine.get_player_index(ip)
    ok = engine.reorder_card(p, idx)
  return jsonify({'ok': ok})


@app.route('/draw', methods=['POST'])
def draw_card():
  ip = get_client_ip()
  source = get_json_body().get('source')
  with game_lock:
    p = engine.get_player_index(ip)
    result = engine.draw(p, source)
  return jsonify(result)


@app.route('/discard', methods=['POST'])
def discard_card():
  ip = get_client_ip()
  idx = get_json_body().get('index')
  with game_lock:
    p = engine.get_player_index(ip)
    result = engine.discard(p, idx, by_index=True)
  return jsonify(result)


# --- BROWSER UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Family Rummy</title>
  <style>
    body { font-family: system-ui, sans-serif; text-align: center; background: #1b4d3e; color: white; padding: 20px; padding-top: 50px; }
    .box { background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; max-width: 600px; margin: 15px auto; }
    .card { display: inline-block; background: white; color: black; font-size: 24px; font-weight: bold;
            padding: 18px 14px; margin: 6px; border-radius: 10px; cursor: pointer; min-width: 45px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); user-select: none; }
    .card-sm { font-size: 18px; padding: 10px 9px; margin: 4px; min-width: 32px; cursor: default; }
    .card-locked { opacity: 0.45; cursor: not-allowed; border: 2px dashed #ffd166; }
    .red { color: #d90429; }
    .deck { background: #2b2d42; color: white; border: 2px solid #8d99ae; }
    .set-sep { display: inline-block; width: 18px; }
    input[type="text"] { font-size: 16px; padding: 10px 14px; border-radius: 8px; border: none; margin: 5px; width: 180px; }
    button { font-size: 16px; padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer; margin: 5px; font-weight: bold; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-green { background: #2a9d8f; color: white; }
    .btn-gold { background: #e9c46a; color: black; }
    .btn-stop { position: fixed; top: 14px; right: 14px; background: #e63946; color: white; font-size: 13px; padding: 8px 14px; }
    .status { font-size: 20px; font-weight: bold; color: #ffd166; margin: 10px 0; }
    .badge { background: #2a9d8f; padding: 4px 10px; border-radius: 20px; font-size: 13px; }
    .hint { font-size: 13px; color: #ffd166; margin-top: 6px; min-height: 18px; }
  </style>
</head>
<body>
  <button id="stopGameBtn" class="btn-stop" style="display: none;" onclick="confirmStopGame()">Stop Game</button>

  <h2>🃏 Two-Sets-of-3 Card Game <span id="datasetBadge" class="badge">0 games</span></h2>

  <!-- LOBBY VIEW (Shown before game starts) -->
  <div id="lobbyBox" class="box">
    <h3>Game Lobby (Yet to Start)</h3>
    <p id="lobbyInfo">Players joined: 0 / 4</p>
    <p id="lobbyPlayers" style="color: #a8dadc; font-weight: bold;"></p>
    <p id="myIdentity" style="font-size: 18px; color: #ffd166;"></p>
    <div id="joinControls">
      <input type="text" id="nameInput" placeholder="Enter your name" maxlength="16">
      <button id="joinBtn" class="btn-green" onclick="joinWithName()">Join Game</button>
    </div>
    <button id="startBtn" class="btn-gold" onclick="post('/start')" disabled>Start Game (Need 2+)</button>
  </div>

  <!-- ACTIVE GAME VIEW (Shown once game starts) -->
  <div id="gameArea" style="display: none;">
    <div class="box">
      <div id="lockedPlayerBadge" style="font-size: 18px; font-weight: bold; margin-bottom: 8px;"></div>
      <div id="log" style="color: #a8dadc;"></div>
      <div id="status" class="status"></div>
      <div id="winningHandBox" style="display: none; margin: 12px 0; padding: 10px; background: rgba(255,255,255,0.12); border-radius: 10px;">
        <div id="winningHandLabel" style="font-size: 14px; color: #ffd166; margin-bottom: 6px;"></div>
        <div id="winningHandCards"></div>
      </div>
      <div id="rematchBox" style="display: none; margin: 10px 0;">
        <button class="btn-gold" onclick="post('/rematch')">Play Again (Same Players)</button>
      </div>

      <div style="margin: 20px 0;">
        <div class="card deck" onclick="post('/draw', {source: 'T'})">
          Hidden<br><span id="deckCount" style="font-size:14px;"></span>
        </div>
        <div id="shownCard" class="card" onclick="post('/draw', {source: 'S'})">?</div>
      </div>
    </div>

    <div class="box">
      <div id="handPrompt">Your Hand (Tap a card to move it to the front of your hand):</div>
      <div id="hand"></div>
      <div id="actionHint" class="hint"></div>
    </div>
  </div>

  <script>
    // Preserve ?mock_ip= if testing multiple tabs on one laptop
    const queryStr = window.location.search;
    let wasMyTurn = false;
    let isSubmitting = false;

    function notifyMyTurn() {
      if (navigator.vibrate) {
        navigator.vibrate([200, 100, 200]);
      }
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.4);
      } catch (e) {}
    }

    function colorClass(label) {
      return (label.includes('♥') || label.includes('♦')) ? 'red' : '';
    }

    async function post(url, body = {}) {
      if (isSubmitting) return;
      isSubmitting = true;
      try {
        const res = await fetch(url + queryStr, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const result = await res.json();
        const hintEl = document.getElementById('actionHint');
        if (hintEl) {
          hintEl.innerText = (!result.ok && result.error) ? result.error : '';
        }
      } finally {
        isSubmitting = false;
      }
      refresh();
    }

    function joinWithName() {
      const name = document.getElementById('nameInput').value.trim();
      post('/join', {name: name});
    }

    function confirmStopGame() {
      if (confirm("Are you sure you want to stop the game and return everyone to the lobby?")) {
        post('/restart');
      }
    }

    async function refresh() {
      const res = await fetch('/state' + queryStr);
      const data = await res.json();

      document.getElementById('datasetBadge').innerText = `${data.saved_games} games`;
      document.getElementById('stopGameBtn').style.display = data.started ? 'inline-block' : 'none';

      if (!data.started) {
        document.getElementById('lobbyBox').style.display = 'block';
        document.getElementById('gameArea').style.display = 'none';
        document.getElementById('lobbyInfo').innerText = `Players joined: ${data.joined_count} / 4 — ${data.log}`;
        document.getElementById('lobbyPlayers').innerText = data.player_names.length
          ? `In lobby: ${data.player_names.join(', ')}`
          : '';

        const joinControls = document.getElementById('joinControls');
        const joinBtn = document.getElementById('joinBtn');
        const idText = document.getElementById('myIdentity');
        if (data.my_player !== null) {
          idText.innerText = `✅ You joined as: ${data.my_name}`;
          joinControls.style.display = 'none';
        } else {
          idText.innerText = `Not joined yet (${data.my_ip})`;
          joinControls.style.display = 'block';
          joinBtn.disabled = data.joined_count >= 4;
        }

        const startBtn = document.getElementById('startBtn');
        startBtn.disabled = data.joined_count < 2;
        startBtn.innerText = data.joined_count >= 2 ? `Start Game (${data.joined_count} Players)` : 'Start Game (Need 2+)';
        return;
      }

      document.getElementById('lobbyBox').style.display = 'none';
      document.getElementById('gameArea').style.display = 'block';

      const badge = document.getElementById('lockedPlayerBadge');
      badge.innerText = data.my_player !== null
        ? `🔒 Playing as ${data.my_name}`
        : `👀 Spectating (Game already started)`;

      document.getElementById('log').innerText = data.log;
      document.getElementById('deckCount').innerText = `(${data.deck_count} left)`;

      const shownEl = document.getElementById('shownCard');
      shownEl.innerText = data.shown;
      shownEl.className = 'card ' + colorClass(data.shown);

      const statusEl = document.getElementById('status');
      const winBox = document.getElementById('winningHandBox');
      const rematchBox = document.getElementById('rematchBox');
      const isMyTurn = Boolean(data.started && !data.winner && data.turn === data.my_player);
      if (isMyTurn && !wasMyTurn) {
        notifyMyTurn();
      }
      wasMyTurn = isMyTurn;

      rematchBox.style.display = data.winner ? 'block' : 'none';

      if (data.winner === 'Draw') {
        statusEl.innerText = "🤝 Draw Game (Deck Empty)";
        winBox.style.display = 'none';
      } else if (data.winner) {
        statusEl.innerText = `🏆 ${data.winner} Wins!`;
        winBox.style.display = 'block';
        document.getElementById('winningHandLabel').innerText = `${data.winner}'s Winning Hand:`;
        const winCardsEl = document.getElementById('winningHandCards');
        winCardsEl.innerHTML = '';
        const sets = (data.winning_sets && data.winning_sets.length === 2)
          ? data.winning_sets
          : [data.winning_hand];
        sets.forEach((cardSet, setIdx) => {
          if (setIdx > 0) {
            const sep = document.createElement('span');
            sep.className = 'set-sep';
            winCardsEl.appendChild(sep);
          }
          cardSet.forEach((label) => {
            const div = document.createElement('div');
            div.className = 'card card-sm ' + colorClass(label);
            div.innerText = label;
            winCardsEl.appendChild(div);
          });
        });
      } else {
        winBox.style.display = 'none';
        if (data.turn === data.my_player) {
          statusEl.innerText = data.phase === 'DRAW'
            ? "👉 YOUR TURN: Pick Hidden or Shown card!"
            : "👉 YOUR TURN: Tap a card in your hand to put down!";
        } else {
          statusEl.innerText = `⏳ Waiting for ${data.turn_name}...`;
        }
      }

      const canDiscard = Boolean(!data.winner && data.turn === data.my_player && data.phase === 'DISCARD');
      document.getElementById('handPrompt').innerText = canDiscard
        ? "Your Hand (Tap a card to put it down):"
        : "Your Hand (Tap a card to move it to the front of your hand):";

      const handEl = document.getElementById('hand');
      handEl.innerHTML = '';
      data.hand.forEach((label, idx) => {
        const div = document.createElement('div');
        const isLocked = canDiscard && data.forbidden_discard_idx === idx;
        div.className = 'card ' + colorClass(label) + (isLocked ? ' card-locked' : '');
        div.innerText = label;
        if (isLocked) {
          div.title = 'Just picked from Shown — pick a different card to put down';
        }
        div.onclick = () => {
          if (isLocked) {
            document.getElementById('actionHint').innerText =
              'Cannot put down the exact card you just picked from Shown.';
            return;
          }
          canDiscard ? post('/discard', {index: idx}) : post('/reorder', {index: idx});
        };
        handEl.appendChild(div);
      });
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=7171, debug=True)
