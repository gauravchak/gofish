"""Flask web server for Two-Sets-of-3 (6-card Family Rummy) with Computer Players."""

import json
import os
import random
import threading
import time

from flask import Flask, jsonify, render_template, request

from bots import BOT_OPTIONS, BOT_STARTING_ELO, MODELS_DIR, choose_bot_for_elo, create_bot
from bots.heuristic import get_next_player_hints, get_unseen_mask, score_hand
from game import FamilyStore, GameEngine, TrajectoryLogger
from game.rules import card_label, card_to_id, id_to_card, sort_cards
from scripts.evaluate import play_match

app = Flask(__name__)

LOG_FILE = 'trajectories.jsonl'
FAMILY_FILE = 'family.json'
LEAGUE_HISTORY_FILE = os.path.join(MODELS_DIR, 'league_history.json')
BOT_SPEED_PRESETS = {'slow': 1.25, 'normal': 0.65, 'fast': 0.18}

game_lock = threading.Lock()
trajectory_logger = TrajectoryLogger(LOG_FILE)
family_store = FamilyStore(FAMILY_FILE)
engine = GameEngine(logger=trajectory_logger, family_store=family_store)

# Maps synthetic bot IDs (e.g., 'bot:1') -> BotPolicy instance.
bot_policies = {}
bot_counter = 0
bot_worker_running = False
bot_speed_key = 'normal'
last_bot_thought = ''


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
  global bot_worker_running, last_bot_thought
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
        step_delay = BOT_SPEED_PRESETS.get(bot_speed_key, 0.65)

      # Pause briefly and compute the bot's decision outside game_lock so the UI
      # stays responsive and humans can clearly see each DRAW and DISCARD step.
      t0 = time.monotonic()
      action = (
          policy.choose_draw(obs)
          if phase == 'DRAW'
          else policy.choose_discard(obs)
      )
      thought = getattr(policy, 'last_thought', '')
      remaining_delay = step_delay - (time.monotonic() - t0)
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
        if thought:
          last_bot_thought = f'{current_player["name"]}: {thought}'
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


def _resolve_bot_key_locked(bot_key):
  """Resolves 'auto' to a difficulty matched to the human players' current Elo."""
  if bot_key != 'auto':
    return bot_key
  human_elos = [
      family_store.get_elo(p['name'])
      for p in engine.players
      if not p['type'].startswith('bot:')
  ]
  target_elo = (
      sum(human_elos) / len(human_elos)
      if human_elos
      else family_store.initial_elo
  )
  return choose_bot_for_elo(target_elo)


def _add_bot_locked(bot_key):
  """Creates and joins a computer player. Caller must hold game_lock."""
  global bot_counter
  resolved_key = _resolve_bot_key_locked(bot_key)
  try:
    policy = create_bot(resolved_key)
  except (ValueError, RuntimeError) as err:
    return {'ok': False, 'error': str(err)}

  bot_counter += 1
  bot_id = f'bot:{bot_counter}'
  existing_names = {p['name'] for p in engine.players}
  display_name = policy.name
  if display_name in existing_names:
    display_name = f'{policy.name[:19]} #{bot_counter}'

  result = engine.join_player(
      bot_id,
      name=display_name,
      player_type=policy.player_type,
      initial_elo=BOT_STARTING_ELO.get(policy.player_type),
  )
  if result['ok']:
    bot_policies[bot_id] = policy
    result['bot_id'] = bot_id
    result['bot_key'] = resolved_key
  return result


# --- WEB ROUTES ---
@app.route('/')
def home():
  return render_template('index.html')


@app.route('/league')
def league_page():
  return render_template('league.html')


@app.route('/league_data')
def league_data():
  payload = {'available': False, 'history': []}
  if os.path.exists(LEAGUE_HISTORY_FILE):
    with open(LEAGUE_HISTORY_FILE, 'r', encoding='utf-8') as f:
      payload = json.load(f)
    payload['available'] = True
  payload['bot_options'] = [o for o in BOT_OPTIONS if o['key'] != 'auto']
  return jsonify(payload)


@app.route('/api/arena_match', methods=['POST'])
def api_arena_match():
  """Runs a fast headless Bot-vs-Bot matchup directly from the /league Lab UI."""
  body = get_json_body()
  key_a = body.get('bot_a', 'rl_alphazero')
  key_b = body.get('bot_b', 'heuristic')
  try:
    games = int(body.get('games', 16))
  except (TypeError, ValueError):
    games = 16
  max_games = 12 if ('tree_2ply' in (key_a, key_b)) else 40
  games = max(2, min(games, max_games))

  try:
    bot_a = create_bot(key_a)
    bot_b = create_bot(key_b)
  except (ValueError, RuntimeError) as err:
    return jsonify({'ok': False, 'error': str(err)})

  wins_a = wins_b = draws = 0
  total_turns = 0
  a_ms, b_ms = [], []
  base_seed = random.randint(1, 100000)

  for i in range(games):
    res = play_match(
        bot_a, bot_b, seed=base_seed + i, logger=None, swap_seats=(i % 2 == 1)
    )
    if res['winner'] == 'A':
      wins_a += 1
    elif res['winner'] == 'B':
      wins_b += 1
    else:
      draws += 1
    total_turns += res['turns']
    a_ms.extend(res['a_ms'])
    b_ms.extend(res['b_ms'])

  return jsonify({
      'ok': True,
      'bot_a_name': bot_a.name,
      'bot_b_name': bot_b.name,
      'games': games,
      'wins_a': wins_a,
      'wins_b': wins_b,
      'draws': draws,
      'avg_turns': round(total_turns / games, 1),
      'avg_a_ms': round(sum(a_ms) / max(len(a_ms), 1), 2),
      'avg_b_ms': round(sum(b_ms) / max(len(b_ms), 1), 2),
  })


@app.route('/api/random_puzzle')
def api_random_puzzle():
  """Deals a fun 6-card hand + 1 Shown card for the Interactive Bot Brain Sandbox."""
  deck = list(range(52))
  random.shuffle(deck)
  hand_ids = sorted(deck[:6])
  shown_id = deck[6]
  return jsonify({
      'ok': True,
      'hand_ids': hand_ids,
      'hand_labels': [card_label(id_to_card(c)) for c in hand_ids],
      'shown_id': shown_id,
      'shown_label': card_label(id_to_card(shown_id)),
  })


def _score_discard_options(bot, obs_disc):
  """Returns sorted list of {'card', 'pct', 'display'} for the Bot Brain bar chart."""
  legal_ids = list(obs_disc['legal_action_ids'])
  hand7 = obs_disc['hand_ids']

  # PyTorch Policy Gradient / X-Ray GRPO / Rollout Distilled: exact masked softmax probs!
  if bot.player_type in (
      'bot:rl_policy_gradient',
      'bot:rl_grpo_xray',
      'bot:rl_distilled',
  ):
    import torch
    from ml.features import extract_all_action_features

    _, all_feats = extract_all_action_features(obs_disc)
    net = (
        bot.discard_net
        if bot.player_type == 'bot:rl_distilled'
        else bot.discard_actor
    )
    with torch.no_grad():
      logits = net(torch.tensor(all_feats, dtype=torch.float32))
      probs = torch.softmax(logits, dim=-1).tolist()
    items = [
        {'card': card_label(id_to_card(cid)), 'raw': p, 'display': f'{100*p:.0f}%'}
        for cid, p in zip(legal_ids, probs)
    ]
  # PyTorch Q-Learning: explicit Q(s, a) in (-1, 1)
  elif bot.player_type == 'bot:rl_q_learning':
    from ml.features import extract_all_action_features

    _, all_feats = extract_all_action_features(obs_disc)
    q_vals = bot.discard_net.predict_batch(all_feats)
    items = [
        {'card': card_label(id_to_card(cid)), 'raw': (q + 1.0) / 2.0, 'display': f'Q={q:+.2f}'}
        for cid, q in zip(legal_ids, q_vals)
    ]
  # PyTorch TD Value & AlphaZero-Lite: predicted win probabilities!
  elif bot.player_type in ('bot:rl_td_value', 'bot:rl_alphazero'):
    from ml.features import extract_afterstate_features

    v_net = bot.value_bot.net if bot.player_type == 'bot:rl_alphazero' else bot.net
    unseen_mask = get_unseen_mask(obs_disc)
    nw, nd = get_next_player_hints(obs_disc)
    rows = []
    for cid in legal_ids:
      idx = hand7.index(cid)
      h6 = hand7[:idx] + hand7[idx + 1 :]
      rows.append(
          extract_afterstate_features(
              h6, unseen_mask, obs_disc['deck_remaining'], nw, nd, discarded_id=cid, took_shown=True
          )
      )
    vals = v_net.predict_batch(rows)
    items = [
        {'card': card_label(id_to_card(cid)), 'raw': v, 'display': f'{100*v:.0f}% win'}
        for cid, v in zip(legal_ids, vals)
    ]
  else:
    # Heuristic / Tree / Beginner / Random: show hand score for each candidate discard.
    unseen_mask = get_unseen_mask(obs_disc)
    raw_scores = []
    for cid in legal_ids:
      idx = hand7.index(cid)
      h6 = hand7[:idx] + hand7[idx + 1 :]
      raw_scores.append(score_hand(h6, unseen_mask))
    hi = max(max(raw_scores), 1.0)
    items = [
        {'card': card_label(id_to_card(cid)), 'raw': sc / hi, 'display': f'{sc:.1f}'}
        for cid, sc in zip(legal_ids, raw_scores)
    ]

  items.sort(key=lambda x: x['raw'], reverse=True)
  top = max(max(x['raw'] for x in items), 1e-6)
  for item in items:
    item['bar_pct'] = round(100.0 * max(item['raw'], 0.0) / top, 1)
  return items


@app.route('/api/ask_bot', methods=['POST'])
def api_ask_bot():
  """Asks any computer player what it would draw & discard on a given 6-card hand."""
  body = get_json_body()
  bot_key = body.get('bot_key', 'rl_alphazero')
  hand_ids = body.get('hand_ids') or []
  shown_id = body.get('shown_id')

  if (
      not isinstance(hand_ids, list)
      or len(hand_ids) != 6
      or not isinstance(shown_id, int)
      or len(set(hand_ids + [shown_id])) != 7
      or any(not (0 <= c < 52) for c in hand_ids + [shown_id])
  ):
    return jsonify({'ok': False, 'error': 'Provide 6 unique cards + 1 unique Shown card.'})

  try:
    bot = create_bot(bot_key)
  except (ValueError, RuntimeError) as err:
    return jsonify({'ok': False, 'error': str(err)})

  # Build a clean single-turn GameEngine state to evaluate both DRAW and post-Shown DISCARD.
  sim = GameEngine(logger=None)
  sim.join_player('b0', bot.name, player_type=bot.player_type)
  sim.join_player('b1', 'Opponent', player_type='human')
  sim.start_match(seed=1)

  used = set(hand_ids + [shown_id])
  remaining = [id_to_card(c) for c in range(52) if c not in used]
  sim.hands[0] = [id_to_card(c) for c in sorted(hand_ids)]
  sim.hands[1] = remaining[:6]
  sim.deck = remaining[6:]
  sim.shown = [id_to_card(shown_id)]

  obs_draw = sim.get_observation(0)
  draw_choice = bot.choose_draw(obs_draw)
  draw_thought = bot.last_thought or (
      'Would pick Shown' if draw_choice == 'S' else 'Would draw Hidden'
  )

  # Also simulate taking Shown so kids can see how the bot ranks every discard!
  sim.draw(0, 'S')
  obs_disc = sim.get_observation(0)
  disc_card = bot.choose_discard(obs_disc)
  disc_thought = bot.last_thought or f'Would put down {card_label(disc_card)}'
  ranking = _score_discard_options(bot, obs_disc)

  return jsonify({
      'ok': True,
      'bot_name': bot.name,
      'draw_choice': draw_choice,
      'draw_label': f'Shown ({card_label(id_to_card(shown_id))})' if draw_choice == 'S' else 'Hidden Deck',
      'draw_thought': draw_thought,
      'discard_choice': card_label(disc_card),
      'discard_thought': disc_thought,
      'discard_ranking': ranking,
  })


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
  bot_key = get_json_body().get('bot_key', 'auto')
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
  bot_key = body.get('bot_key', 'auto')

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


@app.route('/bot_speed', methods=['POST'])
def bot_speed_route():
  global bot_speed_key
  speed = get_json_body().get('speed', 'normal')
  if speed not in BOT_SPEED_PRESETS:
    return jsonify({'ok': False, 'error': 'Invalid speed preset.'})
  with game_lock:
    bot_speed_key = speed
  return jsonify({'ok': True, 'speed': bot_speed_key})


@app.route('/start', methods=['POST'])
def start_route():
  global last_bot_thought
  with game_lock:
    last_bot_thought = ''
    ok = engine.start_match()
  if ok:
    schedule_bot_turn_if_needed()
  return jsonify({'ok': ok})


@app.route('/rematch', methods=['POST'])
def rematch_route():
  global last_bot_thought
  with game_lock:
    last_bot_thought = ''
    ok = engine.rematch()
  if ok:
    schedule_bot_turn_if_needed()
  return jsonify({'ok': ok})


@app.route('/restart', methods=['POST'])
def restart_route():
  global last_bot_thought
  with game_lock:
    engine.reset_lobby()
    bot_policies.clear()
    last_bot_thought = ''
  return jsonify({'ok': True})


@app.route('/state')
def get_state():
  ip = get_client_ip()
  with game_lock:
    state = engine.get_ui_state(ip)
    state['bot_options'] = list(BOT_OPTIONS)
    state['bot_speed'] = bot_speed_key

    # Fairness check: only expose Bot Brain thoughts and open bot hands when
    # EVERY seat is a computer player (Bot-vs-Bot Arena) or after the game ends.
    is_bot_arena = bool(
        engine.started
        and engine.players
        and all(p['type'].startswith('bot:') for p in engine.players)
    )
    state['is_bot_arena'] = is_bot_arena
    state['bot_thought'] = (
        last_bot_thought if (is_bot_arena or engine.winner is not None) else ''
    )
    if is_bot_arena:
      state['arena_hands'] = [
          {
              'name': p['name'],
              'cards': [card_label(c) for c in sort_cards(engine.hands[idx])],
          }
          for idx, p in enumerate(engine.players)
      ]
    else:
      state['arena_hands'] = []
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


if __name__ == '__main__':
  # Never combine Flask's interactive debugger with host='0.0.0.0' on a LAN.
  # Set FLASK_DEBUG=1 only for local single-machine development (binds to 127.0.0.1).
  debug_mode = os.environ.get('FLASK_DEBUG') == '1'
  host = '127.0.0.1' if debug_mode else '0.0.0.0'
  app.run(host=host, port=7171, debug=debug_mode)
