"""Flask web server for Two-Sets-of-3 (6-card Family Rummy) with Computer Players."""

import json
import os
import threading
import time

from flask import Flask, jsonify, render_template, request

from bots import BOT_OPTIONS, BOT_STARTING_ELO, MODELS_DIR, choose_bot_for_elo, create_bot
from game import FamilyStore, GameEngine, TrajectoryLogger
from game.rules import card_label, sort_cards

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
  if not os.path.exists(LEAGUE_HISTORY_FILE):
    return jsonify({'available': False, 'history': []})
  with open(LEAGUE_HISTORY_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
  data['available'] = True
  return jsonify(data)


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
