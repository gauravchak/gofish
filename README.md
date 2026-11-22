# gofish

A family rummy style game as an illustrative journey to teach your kids how to set up a web based game and then build a computer player of it.

---

## The Card Game (Two-Sets-of-3)

- **Players:** 2 to 4 players
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

## Quick Start: Play on Home Wi-Fi (Intranet)

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

### 3. Join from Phones, Tablets, or Laptops on the Same Wi-Fi
1. Find your computer's local Wi-Fi IP address:
   - **macOS:** `ipconfig getifaddr en0` (or System Settings → Wi-Fi → Details)
   - **Windows:** `ipconfig` (look for `IPv4 Address`)
   - **Linux:** `hostname -I`
2. Open a browser on each device and go to:
   ```text
   http://<YOUR-LOCAL-IP>:7171
   ```
3. Enter your names, tap **Join Game**, and hit **Start Game**!

> **Testing solo on one laptop?** Open two browser tabs with different mock IPs:
> - Tab 1: `http://localhost:7171/?mock_ip=alice`
> - Tab 2: `http://localhost:7171/?mock_ip=bob`

---

## How the Code Is Organized

The code is intentionally small and split into clear pieces so you can read one file at a time:

```text
gofish/
├── app.py              # Thin Flask web server + browser UI (talks to GameEngine)
├── game/
│   ├── rules.py        # Pure card math: 0..51 card IDs, valid sets, win checking
│   ├── engine.py       # Headless game state machine (used by humans, bots, and tests)
│   └── logger.py       # Saves finished games to trajectories.jsonl (schema v2)
├── tests/
│   └── test_game.py    # Automated tests for rules, turn flow, and clean logging
└── requirements.txt    # Minimal dependencies (just Flask!)
```

**Why separate `game/` from `app.py`?**
- `app.py` only handles HTTP requests and the web page.
- `game/engine.py` has no web code at all, so a computer player can `clone()` the game to think 1 or 2 moves ahead, and we can run thousands of fast practice games without a browser.

---

## How Playing Creates ML Training Data (`trajectories.jsonl`)

Every time a game finishes, the server appends one JSON line per decision (`DRAW` or `DISCARD`) to `trajectories.jsonl`. Each line answers three questions:

1. **What did the player see *before* acting?** (`hand`, `shown_card`, `discard_pile`, `deck_remaining`, `public_history`, `legal_actions`)
2. **What did the player choose?** (`action` / `action_id`)
3. **How did the game end?** (`won`, `reward`: `+1.0` win / `-1.0` loss / `0.0` draw, `steps_remaining`)

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
- **Human-readable + ML-ready:** Cards are stored both as readable `[rank, suit]` pairs and canonical `0..51` integer IDs (`hand_ids`, `action_id`) so kids can inspect the logs *and* turn them into numbers for a model.

---

## Running Tests

No extra test dependencies required — uses Python's built-in `unittest`:

```bash
python3 -m unittest discover -s tests -v
```

---

## Project Roadmap

- [x] **Step 1 — Playable Intranet Game & Clean Trajectory Logging:** Split rules/engine/logger from Flask, record leak-free v2 trajectories, add tests and setup guide.
- [ ] **Step 2 — Rules & 1-/2-Level Tree Computer Player:** Add a simple, explainable bot that scores hands and looks 1–2 moves ahead, plus a lobby toggle so a single player can play against the computer.
- [ ] **Step 3 — ML Computer Player:** Train a model from collected `trajectories.jsonl` games and plug it into the exact same bot interface so you can play against the AI you trained together!
