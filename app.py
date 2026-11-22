"""Flask web server for Two-Sets-of-3 (6-card Family Rummy) with Computer Players."""

import os
import threading
import time

from flask import Flask, jsonify, render_template_string, request

from bots import BOT_OPTIONS, create_bot
from game import GameEngine, TrajectoryLogger

app = Flask(__name__)

LOG_FILE = 'trajectories.jsonl'
BOT_STEP_DELAY_SECONDS = 0.65

game_lock = threading.Lock()
trajectory_logger = TrajectoryLogger(LOG_FILE)
engine = GameEngine(logger=trajectory_logger)

# Maps synthetic bot IDs (e.g., 'bot:1') -> BotPolicy instance.
bot_policies = {}
bot_counter = 0
bot_worker_running = False


def get_client_ip():
  # Allow ?mock_ip=x only from localhost so you can test multiple tabs on one
  # laptop without letting devices on the LAN spoof another player's IP.
  if request.remote_addr in ('127.0.0.1', '::1'):
    mock_ip = request.args.get('mock_ip')
    if mock_ip:
      return mock_ip
  return request.remote_addr


def get_json_body():
  return request.get_json(silent=True) or {}


def _bot_turn_loop():
  """Background worker that plays computer turns one paced step at a time."""
  global bot_worker_running
  try:
    while True:
      with game_lock:
        if not engine.started or engine.winner is not None:
          return
        current_idx = engine.turn
        current_player = engine.players[current_idx]
        policy = bot_policies.get(current_player['id'])
        if policy is None:
          return

        expected_game_id = engine.game_id
        expected_step = engine.step_count
        phase = engine.phase
        obs = engine.get_observation(current_idx)

      # Pause briefly and compute the bot's decision outside game_lock so the UI
      # stays responsive and humans can clearly see each DRAW and DISCARD step.
      t0 = time.monotonic()
      action = (
          policy.choose_draw(obs)
          if phase == 'DRAW'
          else policy.choose_discard(obs)
      )
      remaining_delay = BOT_STEP_DELAY_SECONDS - (time.monotonic() - t0)
      if remaining_delay > 0:
        time.sleep(remaining_delay)

      with game_lock:
        # Guard against a reset/rematch happening while the bot was thinking.
        if (
            not engine.started
            or engine.winner is not None
            or engine.game_id != expected_game_id
            or engine.step_count != expected_step
        ):
          return

        if phase == 'DRAW':
          engine.draw(current_idx, action)
        else:
          engine.discard(current_idx, action, by_index=False)
  finally:
    with game_lock:
      bot_worker_running = False


def schedule_bot_turn_if_needed():
  """Starts the background bot worker if it is currently a computer player's turn."""
  global bot_worker_running
  with game_lock:
    if (
        bot_worker_running
        or not engine.started
        or engine.winner is not None
        or not engine.players
    ):
      return
    current_id = engine.players[engine.turn]['id']
    if current_id not in bot_policies:
      return
    bot_worker_running = True
    thread = threading.Thread(target=_bot_turn_loop, daemon=True)
    thread.start()


def _add_bot_locked(bot_key):
  """Creates and joins a computer player. Caller must hold game_lock."""
  global bot_counter
  try:
    policy = create_bot(bot_key)
  except ValueError as err:
    return {'ok': False, 'error': str(err)}

  bot_counter += 1
  bot_id = f'bot:{bot_counter}'
  # Give a clean short seat name (e.g., 'Tree Bot (1-Level)' or 'Tree Bot #2').
  existing_names = {p['name'] for p in engine.players}
  display_name = policy.name
  if display_name in existing_names:
    display_name = f'{policy.name[:19]} #{bot_counter}'

  result = engine.join_player(
      bot_id, name=display_name, player_type=policy.player_type
  )
  if result['ok']:
    bot_policies[bot_id] = policy
    result['bot_id'] = bot_id
  return result


# --- WEB API ROUTES ---
@app.route('/')
def home():
  return render_template_string(HTML_PAGE)


@app.route('/join', methods=['POST'])
def join_game():
  ip = get_client_ip()
  raw_name = get_json_body().get('name', '')
  if not isinstance(raw_name, str):
    raw_name = ''
  with game_lock:
    result = engine.join_player(ip, name=raw_name, player_type='human')
  return jsonify(result)


@app.route('/add_bot', methods=['POST'])
def add_bot_route():
  bot_key = get_json_body().get('bot_key', 'tree_1ply')
  with game_lock:
    result = _add_bot_locked(bot_key)
  return jsonify(result)


@app.route('/remove_bot', methods=['POST'])
def remove_bot_route():
  bot_id = get_json_body().get('bot_id', '')
  with game_lock:
    if bot_id not in bot_policies:
      return jsonify({'ok': False, 'error': 'Bot not found.'})
    result = engine.remove_player(bot_id)
    if result['ok']:
      bot_policies.pop(bot_id, None)
  return jsonify(result)


@app.route('/quick_play', methods=['POST'])
def quick_play_route():
  """One-click solo mode: joins you, adds the chosen computer player, and starts."""
  ip = get_client_ip()
  body = get_json_body()
  raw_name = body.get('name', '')
  if not isinstance(raw_name, str):
    raw_name = ''
  bot_key = body.get('bot_key', 'tree_1ply')

  with game_lock:
    if engine.started:
      return jsonify({'ok': False, 'error': 'Game already started.'})
    if engine.get_player_index(ip) is None:
      join_res = engine.join_player(ip, name=raw_name, player_type='human')
      if not join_res['ok']:
        return jsonify(join_res)
    if len(engine.players) < 2:
      bot_res = _add_bot_locked(bot_key)
      if not bot_res['ok']:
        return jsonify(bot_res)
    ok = engine.start_match()

  if ok:
    schedule_bot_turn_if_needed()
  return jsonify({'ok': ok})


@app.route('/start', methods=['POST'])
def start_route():
  with game_lock:
    ok = engine.start_match()
  if ok:
    schedule_bot_turn_if_needed()
  return jsonify({'ok': ok})


@app.route('/rematch', methods=['POST'])
def rematch_route():
  with game_lock:
    ok = engine.rematch()
  if ok:
    schedule_bot_turn_if_needed()
  return jsonify({'ok': ok})


@app.route('/restart', methods=['POST'])
def restart_route():
  with game_lock:
    engine.reset_lobby()
    bot_policies.clear()
  return jsonify({'ok': True})


@app.route('/state')
def get_state():
  ip = get_client_ip()
  with game_lock:
    state = engine.get_ui_state(ip)
    state['bot_options'] = list(BOT_OPTIONS)
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
  if result.get('ok'):
    schedule_bot_turn_if_needed()
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
    .box { background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; max-width: 620px; margin: 15px auto; }
    .card { display: inline-block; background: white; color: black; font-size: 24px; font-weight: bold;
            padding: 18px 14px; margin: 6px; border-radius: 10px; cursor: pointer; min-width: 45px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); user-select: none; }
    .card-sm { font-size: 18px; padding: 10px 9px; margin: 4px; min-width: 32px; cursor: default; }
    .card-locked { opacity: 0.45; cursor: not-allowed; border: 2px dashed #ffd166; }
    .red { color: #d90429; }
    .deck { background: #2b2d42; color: white; border: 2px solid #8d99ae; }
    .set-sep { display: inline-block; width: 18px; }
    input[type="text"], select { font-size: 15px; padding: 10px 12px; border-radius: 8px; border: none; margin: 5px; }
    input[type="text"] { width: 180px; }
    button { font-size: 15px; padding: 10px 16px; border-radius: 8px; border: none; cursor: pointer; margin: 5px; font-weight: bold; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-green { background: #2a9d8f; color: white; }
    .btn-gold { background: #e9c46a; color: black; }
    .btn-blue { background: #457b9d; color: white; }
    .btn-stop { position: fixed; top: 14px; right: 14px; background: #e63946; color: white; font-size: 13px; padding: 8px 14px; }
    .status { font-size: 20px; font-weight: bold; color: #ffd166; margin: 10px 0; }
    .badge { background: #2a9d8f; padding: 4px 10px; border-radius: 20px; font-size: 13px; }
    .hint { font-size: 13px; color: #ffd166; margin-top: 6px; min-height: 18px; }
    .roster { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 10px 0; }
    .chip { background: rgba(255,255,255,0.14); padding: 6px 12px; border-radius: 20px; font-size: 14px; display: inline-flex; align-items: center; gap: 6px; }
    .chip-remove { background: rgba(230,57,70,0.85); color: white; border-radius: 50%; width: 20px; height: 20px; padding: 0; margin: 0; font-size: 12px; line-height: 20px; }
    .bot-panel { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.18); }
    .moves-list { list-style: none; padding: 0; margin: 8px 0 0 0; font-size: 13px; color: #d8f3dc; text-align: left; max-width: 420px; margin-left: auto; margin-right: auto; }
    .moves-list li { padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
  </style>
</head>
<body>
  <button id="stopGameBtn" class="btn-stop" style="display: none;" onclick="confirmStopGame()">Stop Game</button>

  <h2>🃏 Two-Sets-of-3 Card Game <span id="datasetBadge" class="badge">0 games</span></h2>

  <!-- LOBBY VIEW (Shown before game starts) -->
  <div id="lobbyBox" class="box">
    <h3>Game Lobby (Yet to Start)</h3>
    <p id="lobbyInfo">Players joined: 0 / 4</p>
    <div id="lobbyRoster" class="roster"></div>
    <p id="myIdentity" style="font-size: 18px; color: #ffd166;"></p>
    <div id="joinControls">
      <input type="text" id="nameInput" placeholder="Enter your name" maxlength="20">
      <button id="joinBtn" class="btn-green" onclick="joinWithName()">Join Game</button>
    </div>

    <div class="bot-panel">
      <div style="font-size: 14px; color: #a8dadc; margin-bottom: 6px;">🤖 Play with a Computer Player</div>
      <select id="botSelect"></select>
      <button id="addBotBtn" class="btn-blue" onclick="addBot()">+ Add Computer</button>
      <button id="quickPlayBtn" class="btn-gold" onclick="quickPlaySolo()">Quick Play Solo</button>
    </div>

    <div style="margin-top: 12px;">
      <button id="startBtn" class="btn-gold" onclick="post('/start')" disabled>Start Game (Need 2+)</button>
    </div>
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

    <div class="box" id="recentMovesBox" style="display: none;">
      <div style="font-size: 13px; color: #a8dadc; font-weight: bold;">Recent Moves</div>
      <ul id="recentMovesList" class="moves-list"></ul>
    </div>
  </div>

  <script>
    const queryStr = window.location.search;
    let wasMyTurn = false;
    let isSubmitting = false;
    let botOptionsLoaded = false;

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

    function addBot() {
      const botKey = document.getElementById('botSelect').value || 'tree_1ply';
      post('/add_bot', {bot_key: botKey});
    }

    function quickPlaySolo() {
      const name = document.getElementById('nameInput').value.trim();
      const botKey = document.getElementById('botSelect').value || 'tree_1ply';
      post('/quick_play', {name: name, bot_key: botKey});
    }

    function confirmStopGame() {
      if (confirm("Are you sure you want to stop the game and return everyone to the lobby?")) {
        post('/restart');
      }
    }

    function populateBotOptions(options) {
      if (botOptionsLoaded || !options || !options.length) return;
      const select = document.getElementById('botSelect');
      select.innerHTML = '';
      options.forEach((opt) => {
        const el = document.createElement('option');
        el.value = opt.key;
        el.innerText = opt.label;
        if (opt.key === 'tree_1ply') el.selected = true;
        select.appendChild(el);
      });
      botOptionsLoaded = true;
    }

    async function refresh() {
      const res = await fetch('/state' + queryStr);
      const data = await res.json();

      populateBotOptions(data.bot_options);
      document.getElementById('datasetBadge').innerText = `${data.saved_games} games`;
      document.getElementById('stopGameBtn').style.display = data.started ? 'inline-block' : 'none';

      if (!data.started) {
        document.getElementById('lobbyBox').style.display = 'block';
        document.getElementById('gameArea').style.display = 'none';
        document.getElementById('lobbyInfo').innerText = `Players joined: ${data.joined_count} / 4 — ${data.log}`;

        const rosterEl = document.getElementById('lobbyRoster');
        rosterEl.innerHTML = '';
        (data.players || []).forEach((p) => {
          const chip = document.createElement('span');
          chip.className = 'chip';
          const label = document.createElement('span');
          label.innerText = (p.is_bot ? '🤖 ' : '🧑 ') + p.name;
          chip.appendChild(label);
          if (p.is_bot) {
            const rm = document.createElement('button');
            rm.className = 'chip-remove';
            rm.innerText = '×';
            rm.title = 'Remove computer player';
            rm.onclick = () => post('/remove_bot', {bot_id: p.id});
            chip.appendChild(rm);
          }
          rosterEl.appendChild(chip);
        });

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

        document.getElementById('addBotBtn').disabled = data.joined_count >= 4;
        document.getElementById('quickPlayBtn').style.display =
          (data.joined_count === 0 || (data.joined_count === 1 && data.my_player !== null))
            ? 'inline-block' : 'none';

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

      const moves = data.recent_moves || [];
      const movesBox = document.getElementById('recentMovesBox');
      const movesList = document.getElementById('recentMovesList');
      if (moves.length > 0) {
        movesBox.style.display = 'block';
        movesList.innerHTML = '';
        moves.slice().reverse().forEach((entry) => {
          const li = document.createElement('li');
          li.innerText = entry;
          movesList.appendChild(li);
        });
      } else {
        movesBox.style.display = 'none';
      }
    }

    setInterval(refresh, 750);
    refresh();
  </script>
</body>
</html>
"""


if __name__ == '__main__':
  # Never combine Flask's interactive debugger with host='0.0.0.0' on a LAN.
  # Set FLASK_DEBUG=1 only for local single-machine development (binds to 127.0.0.1).
  debug_mode = os.environ.get('FLASK_DEBUG') == '1'
  host = '127.0.0.1' if debug_mode else '0.0.0.0'
  app.run(host=host, port=7171, debug=debug_mode)
