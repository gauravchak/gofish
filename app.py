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
  ranks = sorted(c[0] for c in three_cards)
  suits = [c[1] for c in three_cards]
  if ranks[0] == ranks[1] == ranks[2]:
    return True
  if len(set(suits)) == 1 and ranks[0] + 1 == ranks[1] and ranks[1] + 1 == ranks[2]:
    return True
  return False


def check_win(hand):
  if len(hand) != 6:
    return False
  for set1 in combinations(hand, 3):
    set2 = [c for c in hand if c not in set1]
    if is_valid_set(set1) and is_valid_set(set2):
      return True
  return False


# --- 2. GAME STATE & TRAJECTORY RECORDER ---
game = {}
LOG_FILE = 'trajectories.jsonl'


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


def reset_game(num_players=2):
  deck = [[r, s] for s in SUITS for r in RANKS]  # Stored as [rank, suit] lists for clean JSON
  random.shuffle(deck)
  game.clear()
  game.update({
      'game_id': str(uuid.uuid4())[:8],
      'num_players': num_players,
      'deck': deck,
      'hands': [[deck.pop() for _ in range(6)] for _ in range(num_players)],
      'shown': [deck.pop()],
      'turn': 0,
      'step_count': 0,
      'phase': 'DRAW',
      'winner': None,
      'saved': False,
      'trajectory': [],
      'log': 'Game started! Player 1 goes first.',
  })


reset_game(2)

# --- 3. WEB API ROUTES ---
@app.route('/')
def home():
  return render_template_string(HTML_PAGE)


@app.route('/state')
def get_state():
  p = int(request.args.get('player', 0))
  hand = sorted(game['hands'][p], key=lambda c: (c[1], c[0]))
  game['hands'][p] = hand

  # Count how many games we have recorded so far
  saved_games = 0
  try:
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
      saved_games = len({json.loads(line)['game_id'] for line in f if line.strip()})
  except FileNotFoundError:
    pass

  return jsonify({
      'num_players': game['num_players'],
      'hand': [card_label(c) for c in hand],
      'shown': card_label(game['shown'][-1]) if game['shown'] else 'Empty',
      'deck_count': len(game['deck']),
      'turn': game['turn'],
      'phase': game['phase'],
      'winner': game['winner'],
      'log': game['log'],
      'saved_games': saved_games,
  })


@app.route('/draw', methods=['POST'])
def draw_card():
  data = request.json
  p = data['player']
  source = data['source']  # 'T' or 'S'

  if game['winner'] is not None or game['turn'] != p or game['phase'] != 'DRAW':
    return jsonify({'ok': False})

  shown_before = list(game['shown'][-1]) if game['shown'] else None
  hand_before = [list(c) for c in sorted(game['hands'][p], key=lambda c: (c[1], c[0]))]

  if source == 'S' and game['shown']:
    card = game['shown'].pop()
  elif game['deck']:
    card = game['deck'].pop()
  else:
    return jsonify({'ok': False})

  # Record Decision (a): Pick T or S
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
      'action': source,  # 'T' or 'S'
  })
  game['step_count'] += 1

  game['hands'][p].append(card)
  game['phase'] = 'DISCARD'
  game['log'] = f'Player {p+1} picked from {"Shown" if source == "S" else "Hidden Deck"}.'
  return jsonify({'ok': True})


@app.route('/discard', methods=['POST'])
def discard_card():
  data = request.json
  p = data['player']
  idx = data['index']

  if (
      game['winner'] is not None
      or game['turn'] != p
      or game['phase'] != 'DISCARD'
  ):
    return jsonify({'ok': False})

  hand_before = [list(c) for c in game['hands'][p]]
  card = game['hands'][p].pop(idx)

  # Record Decision (b): Which card to put down
  game['trajectory'].append({
      'timestamp': datetime.utcnow().isoformat(),
      'game_id': game['game_id'],
      'num_players': game['num_players'],
      'step': game['step_count'],
      'player': p,
      'phase': 'DISCARD',
      'hand': hand_before,  # All 7 cards before discarding
      'shown_card': list(game['shown'][-1]) if game['shown'] else None,
      'discard_history': [list(c) for c in game['shown']],
      'deck_remaining': len(game['deck']),
      'action': list(card),  # e.g. [7, '♥']
  })
  game['step_count'] += 1

  game['shown'].append(card)
  game['log'] = f'Player {p+1} put down {card_label(card)}.'

  if check_win(game['hands'][p]):
    game['winner'] = p + 1
    game['log'] = f'🎉 PLAYER {p+1} WINS! (Trajectory saved)'
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


@app.route('/restart', methods=['POST'])
def restart():
  num = int(request.json.get('players', 2))
  reset_game(num)
  return jsonify({'ok': True})


# --- 4. BROWSER UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Family Rummy + Data Collector</title>
  <style>
    body { font-family: system-ui, sans-serif; text-align: center; background: #1b4d3e; color: white; padding: 20px; }
    .box { background: rgba(0,0,0,0.25); padding: 15px; border-radius: 12px; max-width: 600px; margin: 15px auto; }
    .card { display: inline-block; background: white; color: black; font-size: 24px; font-weight: bold;
            padding: 18px 14px; margin: 6px; border-radius: 10px; cursor: pointer; min-width: 45px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: transform 0.1s; }
    .card:hover { transform: translateY(-5px); }
    .red { color: #d90429; }
    .deck { background: #2b2d42; color: white; border: 2px solid #8d99ae; }
    button, select { font-size: 16px; padding: 8px 14px; border-radius: 8px; border: none; cursor: pointer; margin: 4px; }
    .status { font-size: 20px; font-weight: bold; color: #ffd166; margin: 10px 0; }
    .badge { background: #2a9d8f; padding: 4px 10px; border-radius: 20px; font-size: 13px; }
  </style>
</head>
<body>
  <h2>🃏 Two-Sets-of-3 Card Game <span id="datasetBadge" class="badge">0 games recorded</span></h2>
  
  <div class="box">
    I am: 
    <select id="playerSelect" onchange="refresh()">
      <option value="0">Player 1</option>
      <option value="1">Player 2</option>
      <option value="2">Player 3</option>
      <option value="3">Player 4</option>
    </select>
    | New Game:
    <button onclick="restart(2)">2 Players</button>
    <button onclick="restart(3)">3 Players</button>
    <button onclick="restart(4)">4 Players</button>
  </div>

  <div class="box">
    <div id="log" style="color: #a8dadc;"></div>
    <div id="status" class="status">Loading...</div>
    
    <div style="margin: 20px 0;">
      <div class="card deck" onclick="draw('T')">
        Hidden<br><span id="deckCount" style="font-size:14px;"></span>
      </div>
      <div id="shownCard" class="card" onclick="draw('S')">?</div>
    </div>
    <div style="font-size: 13px; opacity: 0.8;">Tap Hidden Deck (T) or Shown Card (S) to draw</div>
  </div>

  <div class="box">
    <div>Your Hand (Tap a card to put it down):</div>
    <div id="hand"></div>
  </div>

  <script>
    function colorClass(label) {
      return (label.includes('♥') || label.includes('♦')) ? 'red' : '';
    }

    async function refresh() {
      const p = document.getElementById('playerSelect').value;
      const res = await fetch('/state?player=' + p);
      const data = await res.json();

      document.getElementById('datasetBadge').innerText = `${data.saved_games} games recorded`;
      document.getElementById('log').innerText = data.log;
      document.getElementById('deckCount').innerText = `(${data.deck_count} left)`;
      
      const shownEl = document.getElementById('shownCard');
      shownEl.innerText = data.shown;
      shownEl.className = 'card ' + colorClass(data.shown);

      const statusEl = document.getElementById('status');
      if (data.winner === 'Draw') {
        statusEl.innerText = "🤝 Draw Game (Deck Empty)";
      } else if (data.winner) {
        statusEl.innerText = `🏆 Player ${data.winner} Wins!`;
      } else if (data.turn == p) {
        statusEl.innerText = data.phase === 'DRAW' 
          ? "👉 YOUR TURN: Pick Hidden or Shown card!" 
          : "👉 YOUR TURN: Tap a card in your hand to put down!";
      } else {
        statusEl.innerText = `⏳ Waiting for Player ${data.turn + 1}...`;
      }

      const handEl = document.getElementById('hand');
      handEl.innerHTML = '';
      data.hand.forEach((label, idx) => {
        const div = document.createElement('div');
        div.className = 'card ' + colorClass(label);
        div.innerText = label;
        div.onclick = () => discard(idx);
        handEl.appendChild(div);
      });
    }

    async function draw(source) {
      const p = parseInt(document.getElementById('playerSelect').value);
      await fetch('/draw', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({player: p, source: source})
      });
      refresh();
    }

    async function discard(index) {
      const p = parseInt(document.getElementById('playerSelect').value);
      await fetch('/discard', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({player: p, index: index})
      });
      refresh();
    }

    async function restart(players) {
      await fetch('/restart', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({players: players})
      });
      refresh();
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=7171, debug=True)
