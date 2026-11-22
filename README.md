# gofish

A family rummy style game as an illustrative journey to teach your kids how to set up a web based game and then build a computer player of it.

---

## The Card Game (Two-Sets-of-3)

- **Players:** 2 to 4 players (any mix of humans and computer players!)
- **Hand size:** Each player holds **6 cards**.
- **Goal:** Be the first player whose 6 cards form **two sets of 3**:
  - **Three of a kind:** 3 cards of the same rank (e.g., `9♥ 9♦ 9♠`)
  - **Suited sequence:** 3 consecutive cards of the same suit (e.g., `A♣ 2♣ 3♣`, `4♥ 5♥ 6♥`, or `Q♠ K♠ A♠`; wrap-around `K-A-2` does not count)
- **On your turn:**
  1. **Draw** 1 card from either the **Hidden Deck** (`T`) or the face-up **Shown Pile** (`S`).
  2. **Discard** 1 card from your 7 cards onto the Shown Pile.
     - *House rule:* If you picked from the **Shown Pile**, you cannot put that exact same card right back down on the same turn.
- **Game end:** If your remaining 6 cards form two valid sets of 3, you win! If the Hidden Deck runs out first, the game is a draw.

---

## Quick Start: Play Solo or on Home Wi-Fi (Intranet)

### 1. Install Flask
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python3 app.py
```

### 3. Play!
- **Play solo vs the computer:** Open `http://localhost:7171`, type your name, pick a computer difficulty (`Easy`, `Medium`, or `Hard`), and click **Quick Play Solo** (or add 1–3 computer players to the lobby).
- **Play with family on the same Wi-Fi:**
  1. Find your computer's local Wi-Fi IP address:
     - **macOS:** `ipconfig getifaddr en0` (or System Settings → Wi-Fi → Details)
     - **Windows:** `ipconfig` (look for `IPv4 Address`)
     - **Linux:** `hostname -I`
  2. Open `http://<YOUR-LOCAL-IP>:7171` on each phone, tablet, or laptop.
  3. Enter your names, tap **Join Game** (and optionally **+ Add Computer**), then hit **Start Game**!

> **Testing 2 human tabs on one laptop?** Use `?mock_ip=` (only honored from `localhost` so devices on Wi-Fi cannot spoof each other):
> - Tab 1: `http://localhost:7171/?mock_ip=alice`
> - Tab 2: `http://localhost:7171/?mock_ip=bob`

> **Safety note (Trusted Home Wi-Fi Only):** This app uses plain HTTP with no login accounts and is meant for trusted home networks—not public Wi-Fi or the open internet. Flask's interactive debugger is disabled by default when listening on your LAN (`0.0.0.0`); for local auto-reload while coding alone, run `FLASK_DEBUG=1 python3 app.py` (which binds to `127.0.0.1` only).

---

## How the Code Is Organized

The code is intentionally small and split into clear pieces so you can read one file at a time:

```text
gofish/
├── app.py                  # Thin Flask web server + browser UI + paced bot turns
├── game/
│   ├── rules.py            # Pure card math: 0..51 card IDs, valid sets, win checking
│   ├── engine.py           # Headless game state machine (used by humans, bots, and tests)
│   ├── logger.py           # Saves finished games to trajectories.jsonl (schema v2)
│   └── family.py           # Local family.json player profiles & Elo rating leaderboard
├── bots/
│   ├── base.py             # Shared BotPolicy interface (used by rules, tree, and future ML!)
│   ├── heuristic.py        # Rules-based (0-ply) hand scorer using all 100 valid 3-card sets
│   └── tree_bot.py         # 1-level & 2-level Expectimax lookahead tree player
├── scripts/
│   └── evaluate.py         # Headless bot-vs-bot arena (win rates, speed, optional self-play)
├── tests/
│   ├── test_game.py        # Rules, engine turn flow, and zero-leakage trajectory tests
│   ├── test_bots.py        # Bot legality, winning-move, and bot-vs-bot trajectory tests
│   └── test_family.py      # Family profile persistence and Elo math tests
└── requirements.txt        # Minimal dependencies (just Flask!)
```

---

## How the Computer Players Think

All computer players implement the exact same two methods in [bots/base.py](bots/base.py): `choose_draw(obs)` and `choose_discard(obs)`. Later, our **ML player** will plug into the exact same interface!

1. **Heuristic Bot — Easy ([bots/heuristic.py](bots/heuristic.py)):**
   - There are only **100 valid 3-card sets** in a 52-card deck (52 three-of-a-kinds + 48 suited runs).
   - It splits its 6 cards into two groups of 3 (only 10 ways!) and counts how many **live unseen cards** can finish each group—subtracting dead cards already buried in the discard pile.
   - It also watches what the next player picked from `Shown` so it avoids feeding them helpful cards.

2. **1-Level Tree Bot — Medium ([bots/tree_bot.py](bots/tree_bot.py)):**
   - Uses a **1-level Expectimax tree** to think one turn ahead:
     ```text
             [My Choice: Pick Shown ('S') or Hidden ('T')?]
                    /                              \
             Take Shown (known card)         Draw Hidden (chance node)
                   |                                 |
           Best legal discard          Average over every unseen card in deck
                                             -> best discard for each card
     ```
   - On `DISCARD`, it checks every legal discard and averages how strong its hand will be after next turn's random draw.

3. **2-Level Tree Bot — Hard ([bots/tree_bot.py](bots/tree_bot.py)):**
   - Takes the top candidate moves from Level 1 and looks a **second turn ahead** (`Draw 1 → Discard 1 → Draw 2 → Discard 2`), discounting future turns slightly (`gamma = 0.85`) so winning sooner is always preferred.

### Pit Bots Against Each Other (No Browser Needed!)
```bash
python3 -m scripts.evaluate --bot-a heuristic --bot-b random --games 60
python3 -m scripts.evaluate --bot-a tree_1ply --bot-b heuristic --games 60
python3 -m scripts.evaluate --bot-a tree_2ply --bot-b tree_1ply --games 40
```
You can even generate bot self-play training data with `--save-trajectories selfplay.jsonl`!

---

## Local Family Profiles & Elo Leaderboard (`family.json`)

When players join, their name is saved to a local **`family.json`** file (listed in `.gitignore` so personal family names and stats are **never pushed to GitHub**):
- **Quick lobby picker:** Pick a saved family member from the dropdown, or select **`+ New player...`** to type a new name.
- **Automatic Elo ratings (humans + bots!):** Everyone starts at **1000 Elo** (`K = 32`). Beating a higher-rated player (or a harder computer player!) earns more points, and total Elo across all players in a match is conserved (zero-sum) for 2-, 3-, and 4-player games.
- **Live leaderboard:** See everyone's Elo and `W-L-D` record in the lobby—great for tracking whether your family can out-rate the 2-Level Tree Bot (and later, the ML Bot!).

---

## How Playing Creates ML Training Data (`trajectories.jsonl`)

Every finished game appends one JSON line per decision (`DRAW` or `DISCARD`) to `trajectories.jsonl`. Each line answers three questions:

1. **What did the player see *before* acting?** (`hand`, `shown_card`, `discard_pile`, `deck_remaining`, `public_history`, `legal_actions`)
2. **What did the player choose?** (`action` / `action_id`)
3. **How did the game end?** (`won`, `reward`: `+1.0` win / `-1.0` loss / `0.0` draw, `steps_remaining`, `player_type`)

Example `DRAW` step (shortened for readability):
```json
{
  "schema_version": 2,
  "game_id": "a1b2c3d4",
  "step": 0,
  "player": 0,
  "player_type": "human",
  "phase": "DRAW",
  "hand": [[3, "♥"], [9, "♥"], [5, "♦"], [9, "♦"], [1, "♣"], [12, "♣"]],
  "hand_ids": [2, 8, 17, 21, 26, 37],
  "shown_card": [9, "♠"],
  "shown_card_id": 47,
  "discard_pile": [[9, "♠"]],
  "deck_remaining": 39,
  "public_history": [],
  "legal_actions": ["T", "S"],
  "action": "S",
  "action_id": 1,
  "won": true,
  "reward": 1.0,
  "steps_remaining": 14
}
```

Two important ML habits built right in:
- **No peeking into the future (zero leakage):** Every observation is recorded *before* the card is drawn or discarded, and `public_history` never reveals what card another player drew from the hidden deck.
- **Human-readable + ML-ready:** Cards are stored both as readable `[rank, suit]` pairs and canonical `0..51` integer IDs (`hand_ids`, `action_id`), tagged with `player_type` (`"human"`, `"bot:tree_1ply"`, etc.).

---

## Running Tests

No extra test dependencies required — uses Python's built-in `unittest`:

```bash
python3 -m unittest discover -s tests -v
```

---

## Project Roadmap

- [x] **Step 1 — Playable Intranet Game & Clean Trajectory Logging:** Split rules/engine/logger from Flask, record leak-free v2 trajectories, add tests and setup guide.
- [x] **Step 2 — Rules & 1-/2-Level Tree Computer Player:** Add Heuristic, 1-Level Tree, and 2-Level Tree bots, a headless benchmark arena, and solo/mixed lobby play with paced bot turns.
- [ ] **Step 3 — ML Computer Player:** Train a model from collected `trajectories.jsonl` games and plug it into the exact same `BotPolicy` interface so you can play against the AI you trained together!
