from datetime import datetime
from itertools import combinations
import json
import random
import uuid
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# --- 1. GAME LOGIC ---
SUITS = ['♥', '♦', '♣', '♠']
RANKS = list(range(1, 14))  # 1=Ace, 11=J, 12=Q, 13=K


def card_label(card):
  names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
  return f'{names.get(card[0], card[0])}{card[1]}'


def is_valid_set(three_cards):
  """Returns True if 3 cards form either:

    1. Three of a kind (same rank, e.g., 9♥, 9♦, 9♠)
    2. A suited sequence of 3 (e.g., A♣-2♣-3♣, 4♥-5♥-6♥, or Q♣-K♣-A♣)
    """
  if len(three_cards) != 3:
    return False

  ranks = sorted(c[0] for c in three_cards)
  suits = {c[1] for c in three_cards}

  # Case 1: Same number / rank
  if ranks[0] == ranks[1] == ranks[2]:
    return True

  # Case 2: Sequence in the same suit
  if len(suits) == 1:
    # Standard consecutive sequence (e.g., [1, 2, 3] or [11, 12, 13])
    if ranks[0] + 1 == ranks[1] and ranks[1] + 1 == ranks[2]:
      return True
    # Ace-high sequence: Q (12), K (13), A (1)
    if ranks == [1, 12, 13]:
      return True

  return False


def check_win(hand):
  """Returns True if a 6-card hand can be partitioned into two disjoint valid sets of 3."""
  if len(hand) != 6:
    return False

  # Partition using card indices (0..5) so every 3-card split is exact
  all_indices = set(range(6))
  for idx_set1 in combinations(range(6), 3):
    idx_set2 = all_indices - set(idx_set1)
    set1 = [hand[i] for i in idx_set1]
    set2 = [hand[i] for i in idx_set2]
    if is_valid_set(set1) and is_valid_set(set2):
      return True

  return False


# --- 2. GAME STATE & TRAJECTORY RECORDER ---
game = {}
LOG_FILE = 'trajectories.jsonl'


def player_name(idx):
  names = game.get('player_names', [])
  return names[idx] if 0 <= idx < len(names) else f'Player {idx + 1}'


def save_trajectory_to_disk(winner_idx):
  """Backfills the final outcome onto every step and appends to trajectories.jsonl."""
  total_steps = len(game['trajectory'])
  if total_steps == 0:
    return
  with open(LOG_FILE, 'a', encoding='utf-8') as f:
    for i, record in enumerate(game['trajectory']):
      is_winner = (
          (record['player'] == winner_idx) if winner_idx is not None else False
      )
      # Reward: +1.0 for winner, -1.0 for loser, 0.0 for draw
      record['won'] = is_winner
      record['reward'] = (
          0.0 if winner_idx is None else (1.0 if is_winner else -1.0)
      )
      record['steps_remaining'] = total_steps - i
      f.write(json.dumps(record, ensure_ascii=False) + '\n')


def open_lobby():
  """Resets to a 'Yet to Start' lobby where devices can join."""
  game.clear()
  game.update({
      'started': False,
      'players': {},  # Maps IP address -> player index (0, 1, 2, 3)
      'player_names': [],  # List of player names in join order
      'hands': [],
      'deck': [],
      'shown': [],
      'turn': 0,
      'step_count': 0,
      'phase': 'DRAW',
      'winner': None,
      'winning_hand': [],
      'saved': False,
      'trajectory': [],
      'log': 'Waiting for players to join...',
  })


def start_match():
  """Deals cards to everyone currently in the lobby and locks the player list."""
  num_players = len(game['players'])
  if num_players < 2 or game['started']:
    return
  deck = [[r, s] for s in SUITS for r in RANKS]
  random.shuffle(deck)
  game.update({
      'game_id': str(uuid.uuid4())[:8],
      'started': True,
      'num_players': num_players,
      'deck': deck,
      'hands': [[deck.pop() for _ in range(6)] for _ in range(num_players)],
      'shown': [deck.pop()],
      'turn': 0,
      'step_count': 0,
      'phase': 'DRAW',
      'winner': None,
      'winning_hand': [],
      'saved': False,
      'trajectory': [],
      'log': f'Game started with {num_players} players! {player_name(0)} goes first.',
  })


open_lobby()


def get_client_ip():
  # Allows ?mock_ip=x in URL if you ever want to test 2 tabs on the same laptop
  return request.args.get('mock_ip') or request.remote_addr


# --- 3. WEB API ROUTES ---
@app.route('/')
def home():
  return render_template_string(HTML_PAGE)


@app.route('/join', methods=['POST'])
def join_game():
  ip = get_client_ip()
  if (
      not game['started']
      and ip not in game['players']
      and len(game['players']) < 4
  ):
    idx = len(game['players'])
    raw_name = (request.json or {}).get('name', '').strip()
    name = raw_name if raw_name else f'Player {idx + 1}'
    game['players'][ip] = idx
    game['player_names'].append(name)
    game['log'] = f'{name} joined!'
  return jsonify({'ok': True})


@app.route('/start', methods=['POST'])
def start_route():
  start_match()
  return jsonify({'ok': True})


@app.route('/restart', methods=['POST'])
def restart_route():
  open_lobby()
  return jsonify({'ok': True})


@app.route('/state')
def get_state():
  ip = get_client_ip()
  my_player = game['players'].get(
      ip, None
  )  # None if spectator / not joined yet

  saved_games = 0
  try:
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
      saved_games = len(
          {json.loads(line)['game_id'] for line in f if line.strip()}
      )
  except FileNotFoundError:
    pass

  hand = []
  if game['started'] and my_player is not None:
    hand = sorted(game['hands'][my_player], key=lambda c: (c[1], c[0]))
    game['hands'][my_player] = hand

  return jsonify({
      'started': game['started'],
      'joined_count': len(game['players']),
      'player_names': game['player_names'],
      'my_player': my_player,  # Locked to this IP!
      'my_name': player_name(my_player) if my_player is not None else None,
      'my_ip': ip,
      'hand': [card_label(c) for c in hand],
      'shown': card_label(game['shown'][-1]) if game['shown'] else 'Empty',
      'deck_count': len(game['deck']),
      'turn': game['turn'],
      'turn_name': player_name(game['turn']) if game['started'] else '',
      'phase': game['phase'],
      'winner': game['winner'],
      'winning_hand': [card_label(c) for c in game['winning_hand']],
      'log': game['log'],
      'saved_games': saved_games,
  })


@app.route('/draw', methods=['POST'])
def draw_card():
  p = game['players'].get(get_client_ip())
  source = request.json['source']

  if (
      not game['started']
      or p is None
      or game['winner'] is not None
      or game['turn'] != p
      or game['phase'] != 'DRAW'
  ):
    return jsonify({'ok': False})

  shown_before = list(game['shown'][-1]) if game['shown'] else None
  hand_before = [
      list(c) for c in sorted(game['hands'][p], key=lambda c: (c[1], c[0]))
  ]

  if source == 'S' and game['shown']:
    card = game['shown'].pop()
  elif game['deck']:
    card = game['deck'].pop()
  else:
    return jsonify({'ok': False})

  game['trajectory'].append({
      'timestamp': datetime.utcnow().isoformat(),
      'game_id': game['game_id'],
      'num_players': game['num_players'],
      'step': game['step_count'],
      'player': p,
      'phase': 'DRAW',
      'hand': hand_before,
      'shown_card': shown_before,
      'discard_history': [list(c) for c in game['shown']],
      'deck_remaining': len(game['deck']),
      'action': source,
  })
  game['step_count'] += 1

  game['hands'][p].append(card)
  game['phase'] = 'DISCARD'
  game['log'] = (
      f'{player_name(p)} picked from {"Shown" if source == "S" else "Hidden Deck"}.'
  )
  return jsonify({'ok': True})


@app.route('/discard', methods=['POST'])
def discard_card():
  p = game['players'].get(get_client_ip())
  idx = request.json['index']

  if (
      not game['started']
      or p is None
      or game['winner'] is not None
      or game['turn'] != p
      or game['phase'] != 'DISCARD'
  ):
    return jsonify({'ok': False})

  hand_before = [list(c) for c in game['hands'][p]]
  card = game['hands'][p].pop(idx)

  game['trajectory'].append({
      'timestamp': datetime.utcnow().isoformat(),
      'game_id': game['game_id'],
      'num_players': game['num_players'],
      'step': game['step_count'],
      'player': p,
      'phase': 'DISCARD',
      'hand': hand_before,
      'shown_card': list(game['shown'][-1]) if game['shown'] else None,
      'discard_history': [list(c) for c in game['shown']],
      'deck_remaining': len(game['deck']),
      'action': list(card),
  })
  game['step_count'] += 1

  game['shown'].append(card)
  game['log'] = f'{player_name(p)} put down {card_label(card)}.'

  if check_win(game['hands'][p]):
    game['winner'] = player_name(p)
    game['winning_hand'] = sorted(game['hands'][p], key=lambda c: (c[1], c[0]))
    game['log'] = f'🎉 {player_name(p)} WINS! (Trajectory saved)'
    if not game['saved']:
      save_trajectory_to_disk(winner_idx=p)
      game['saved'] = True
  elif len(game['deck']) == 0:
    game['winner'] = 'Draw'
    game['log'] = 'Deck empty — Draw game! (Trajectory saved)'
    if not game['saved']:
      save_trajectory_to_disk(winner_idx=None)
      game['saved'] = True
  else:
    game['turn'] = (game['turn'] + 1) % game['num_players']
    game['phase'] = 'DRAW'
  return jsonify({'ok': True})


# --- 4. BROWSER UI ---
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .card-sm { font-size: 18px; padding: 10px 9px; margin: 4px; min-width: 32px; cursor: default; }
    .red { color: #d90429; }
    .deck { background: #2b2d42; color: white; border: 2px solid #8d99ae; }
    input[type="text"] { font-size: 16px; padding: 10px 14px; border-radius: 8px; border: none; margin: 5px; width: 180px; }
    button { font-size: 16px; padding: 10px 18px; border-radius: 8px; border: none; cursor: pointer; margin: 5px; font-weight: bold; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-green { background: #2a9d8f; color: white; }
    .btn-gold { background: #e9c46a; color: black; }
    .btn-stop { position: fixed; top: 14px; right: 14px; background: #e63946; color: white; font-size: 13px; padding: 8px 14px; }
    .status { font-size: 20px; font-weight: bold; color: #ffd166; margin: 10px 0; }
    .badge { background: #2a9d8f; padding: 4px 10px; border-radius: 20px; font-size: 13px; }
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

      <div style="margin: 20px 0;">
        <div class="card deck" onclick="post('/draw', {source: 'T'})">
          Hidden<br><span id="deckCount" style="font-size:14px;"></span>
        </div>
        <div id="shownCard" class="card" onclick="post('/draw', {source: 'S'})">?</div>
      </div>
    </div>

    <div class="box">
      <div>Your Hand (Tap a card to put it down):</div>
      <div id="hand"></div>
    </div>
  </div>

  <script>
    // Preserve ?mock_ip= if testing multiple tabs on one laptop
    const queryStr = window.location.search;
    let wasMyTurn = false;

    function notifyMyTurn() {
      // 1. Physical buzz (works on Android phones/tablets)
      if (navigator.vibrate) {
        navigator.vibrate([200, 100, 200]);
      }
      // 2. Soft Web Audio chime (works on iPad, Mac, Windows & Android with zero audio files)
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5 note
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15); // A5 note
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
      await fetch(url + queryStr, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
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
        // Show Lobby, hide Game Board
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

      // Game has started: hide Lobby, show locked Game Board
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
      const isMyTurn = Boolean(data.started && !data.winner && data.turn === data.my_player);
      if (isMyTurn && !wasMyTurn) {
        notifyMyTurn();
      }
      wasMyTurn = isMyTurn;

      if (data.winner === 'Draw') {
        statusEl.innerText = "🤝 Draw Game (Deck Empty)";
        winBox.style.display = 'none';
      } else if (data.winner) {
        statusEl.innerText = `🏆 ${data.winner} Wins!`;
        winBox.style.display = 'block';
        document.getElementById('winningHandLabel').innerText = `${data.winner}'s Winning Hand:`;
        const winCardsEl = document.getElementById('winningHandCards');
        winCardsEl.innerHTML = '';
        data.winning_hand.forEach((label) => {
          const div = document.createElement('div');
          div.className = 'card card-sm ' + colorClass(label);
          div.innerText = label;
          winCardsEl.appendChild(div);
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

      const handEl = document.getElementById('hand');
      handEl.innerHTML = '';
      data.hand.forEach((label, idx) => {
        const div = document.createElement('div');
        div.className = 'card ' + colorClass(label);
        div.innerText = label;
        div.onclick = () => post('/discard', {index: idx});
        handEl.appendChild(div);
      });
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


# --- TEST SUITE ---
def run_tests():
  # 1. Your exact hand: A♣, K♣, Q♣ + three 9s (shuffled order!)
  hand_akq_999 = [[1, '♣'], [9, '♥'], [13, '♣'], [9, '♦'], [12, '♣'], [9, '♠']]
  assert check_win(hand_akq_999) is True, 'Failed: A-K-Q clubs + three 9s'

  # 2. Ace-low sequence + three of a kind
  hand_a23_kkk = [[1, '♥'], [2, '♥'], [3, '♥'], [13, '♥'], [13, '♦'], [13, '♠']]
  assert check_win(hand_a23_kkk) is True, 'Failed: A-2-3 hearts + three Kings'

  # 3. Two suited sequences (one Ace-high, one mid-rank)
  hand_two_seqs = [[12, '♠'], [13, '♠'], [1, '♠'], [4, '♦'], [5, '♦'], [6, '♦']]
  assert check_win(hand_two_seqs) is True, 'Failed: Two suited sequences'

  # 4. Two three-of-a-kinds
  hand_two_trips = [[7, '♥'], [7, '♦'], [7, '♣'], [2, '♠'], [2, '♥'], [2, '♣']]
  assert check_win(hand_two_trips) is True, 'Failed: Two three-of-a-kinds'

  # 5. Overlapping cards edge case (e.g., 4♥, 4♦, 4♠, 5♠, 6♠, 4♣ -> valid: {4♥,4♦,4♣} + {4♠,5♠,6♠})
  hand_overlap = [[4, '♥'], [4, '♦'], [4, '♠'], [5, '♠'], [6, '♠'], [4, '♣']]
  assert (
      check_win(hand_overlap) is True
  ), 'Failed: Overlapping rank/sequence split'

  # 6. Invalid: K-A-2 wrap-around should NOT win
  hand_ka2_invalid = [
      [13, '♣'],
      [1, '♣'],
      [2, '♣'],
      [9, '♥'],
      [9, '♦'],
      [9, '♠'],
  ]
  assert (
      check_win(hand_ka2_invalid) is False
  ), 'Failed: K-A-2 wrap-around should be invalid'

  # 7. Invalid: Mixed-suit sequence (A♣, K♦, Q♣)
  hand_mixed_suit = [
      [1, '♣'],
      [13, '♦'],
      [12, '♣'],
      [9, '♥'],
      [9, '♦'],
      [9, '♠'],
  ]
  assert check_win(hand_mixed_suit) is False, 'Failed: Mixed-suit sequence'

  # 8. Invalid: Wrong hand size (7 cards before discarding)
  hand_seven = [
      [1, '♣'],
      [13, '♣'],
      [12, '♣'],
      [9, '♥'],
      [9, '♦'],
      [9, '♠'],
      [3, '♦'],
  ]
  assert check_win(hand_seven) is False, 'Failed: 7-card hand should not win'

  print('✅ All 8 test cases passed!')


if __name__ == '__main__':
  run_tests()
  app.run(host='0.0.0.0', port=7171, debug=True)