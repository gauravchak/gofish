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

## Quick Start: Set Up & Play at Home (Solo or Family Wi-Fi)

**What you need:** One computer (Mac, Windows, or Linux) with **Python 3.10+** installed, plus any phones, tablets, or laptops on the same home Wi-Fi.

### 1. Download the Repo
```bash
git clone https://github.com/gauravchak/gofish.git
cd gofish
```
*(No `git`? Click the green **Code → Download ZIP** button on GitHub, unzip it, and open a terminal inside the `gofish` folder.)*

### 2. Install Flask (and optional PyTorch for RL Bots)
```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-ml.txt   # Optional: enables the 3 PyTorch RL bots & training
```
*(Note: If you skip `requirements-ml.txt`, the game still works cleanly with humans + Random/Beginner/Heuristic/Tree bots! For PyTorch, Python 3.10–3.13 is recommended for prebuilt wheels.)*

### 3. Start the Server
```bash
python3 app.py
```
*(Press `Ctrl+C` in this terminal window whenever you want to stop the server.)*

### 4. Play!
- **Play solo vs the computer:** Open `http://localhost:7171`, type your name (or pick a saved name from the dropdown), leave **Auto-Match by Elo** selected (starts you with a warm-up game against `Random Bot` and levels up as you win!), and click **Quick Play Solo**.
- **Play with family on the same Wi-Fi:**
  1. Find the host computer's local Wi-Fi IP address:
     - **macOS:** `ipconfig getifaddr en0` (or System Settings → Wi-Fi → Details)
     - **Windows:** `ipconfig` (look for `IPv4 Address` under Wi-Fi)
     - **Linux:** `hostname -I`
  2. Open `http://<YOUR-LOCAL-IP>:7171` (for example `http://192.168.1.42:7171`) in a browser on each phone, tablet, or laptop.
  3. Enter or pick your names, tap **Join Game** (and optionally **+ Add Computer**), then hit **Start Game**!

> **Phone or tablet can't connect?**
> - Make sure every device is on the **same Wi-Fi network** (not Guest Wi-Fi, cellular data, or a VPN).
> - If macOS or Windows Firewall pops up asking *"Allow Python to accept incoming network connections?"*, click **Allow**.
> - Double-check you typed `http://` (not `https://`) and `:7171` at the end.

> **Testing 2 human tabs on one laptop?** Use `?mock_ip=` (only honored from `localhost` so devices on Wi-Fi cannot spoof each other):
> - Tab 1: `http://localhost:7171/?mock_ip=alice`
> - Tab 2: `http://localhost:7171/?mock_ip=bob`

> **Safety note (Trusted Home Wi-Fi Only):** This app uses plain HTTP with no login accounts and is meant for trusted home networks—not public Wi-Fi or the open internet. Flask's interactive debugger is disabled by default when listening on your LAN (`0.0.0.0`); for local auto-reload while coding alone, run `FLASK_DEBUG=1 python3 app.py` (which binds to `127.0.0.1` only).

---

## How the Code Is Organized

The code is intentionally small and split into clear pieces so you can read one file at a time:

```text
gofish/
├── app.py                  # Clean Flask web server + paced bot turn worker
├── templates/
│   ├── index.html          # Lobby, game board, Elo leaderboard, & Bot Arena UI
│   └── league.html         # Live SVG Elo-over-Generations chart (/league)
├── game/
│   ├── rules.py            # Pure card math: 0..51 card IDs, valid sets, win checking
│   ├── engine.py           # Headless game state machine (used by humans, bots, and tests)
│   ├── logger.py           # Saves finished games to trajectories.jsonl (schema v2)
│   └── family.py           # Local family.json player profiles & Elo rating leaderboard
├── bots/
│   ├── base.py             # Shared BotPolicy interface (used by rules, tree, and RL bots!)
│   ├── novice.py           # Beginner (Very Easy) bot for welcoming first solo games
│   ├── heuristic.py        # Rules-based (0-ply) hand scorer using all 100 valid 3-card sets
│   └── tree_bot.py         # 1-level & 2-level Expectimax lookahead tree player
├── ml/
│   ├── nn.py               # Clean PyTorch CardMLP (nn.Module, weights_only=True checkpoints)
│   ├── features.py         # Shared 24-feature suit-invariant hand + opponent-memory encoder
│   ├── replay.py           # Experience Replay Buffer for stable mini-batch PyTorch training
│   ├── td_value.py         # RL Bot #1: PyTorch TD(lambda) Value Net with live target network
│   ├── q_learning.py       # RL Bot #2: PyTorch Double DQN with soft target nets & MC blend
│   ├── policy_gradient.py  # RL Bot #3: PyTorch Batched PPO (GAE + clipped surrogate loss)
│   ├── alphazero_bot.py    # RL Bot #4: True AlphaZero-Lite (Policy + Value + 1-Level Expectimax)
│   ├── grpo_bot.py         # RL Bot #5: Omniscient X-Ray Critic + Duplicate-Deal GRPO
│   └── distill_bot.py      # RL Bot #6: Rollout Teacher Distillation (Expert Iteration Student)
├── models/                 # Pre-trained PyTorch checkpoints (models/*.pt) + league_history.json
├── scripts/
│   ├── evaluate.py         # Headless bot-vs-bot arena (win rates, speed, optional self-play)
│   ├── train_rl.py         # 10,000-game progressive Elo RL curriculum trainer
│   ├── league.py           # Generational Bot League (Replay + Hall of Fame + King-of-the-Hill)
│   └── train_variance_reduced.py  # X-Ray GRPO + Rollout Teacher Distillation trainer & benchmark
├── tests/
│   ├── test_game.py        # Rules, engine turn flow, and zero-leakage trajectory tests
│   ├── test_bots.py        # Bot legality, winning-move, and bot-vs-bot trajectory tests
│   ├── test_family.py      # Family profile persistence and Elo math tests
│   └── test_rl.py          # PyTorch CardMLP, feature dims, and all 3 RL bots tests
├── requirements.txt        # Minimal web dependencies (just Flask!)
└── requirements-ml.txt     # Optional ML/RL dependency (PyTorch)
```

---

## How the Computer Players Think

All computer players implement the exact same two methods in [bots/base.py](bots/base.py): `choose_draw(obs)` and `choose_discard(obs)`. Later, our **ML player** will plug into the exact same interface!

By default, **Auto-Match by Elo** picks the right opponent for your rating: brand-new players (1000 Elo) start with a warm-up game against **Random Bot (Starter)** and automatically level up as their Elo climbs (`< 1007` → Random, `< 1035` → Beginner, `< 1085` → Heuristic, `< 1140` → 1-Level Tree, `1140+` → 2-Level Tree).

1. **Random Bot — Starter ([bots/base.py](bots/base.py)):**
   - Picks completely at random among legal moves—a welcoming warm-up opponent for a player's very first solo game.

2. **Beginner Bot — Very Easy ([bots/novice.py](bots/novice.py)):**
   - Plays like someone just learning the game: always grabs an obvious winning card and sometimes protects a complete 3-card set, but otherwise discards casually at random.

3. **Heuristic Bot — Easy ([bots/heuristic.py](bots/heuristic.py)):**
   - There are only **100 valid 3-card sets** in a 52-card deck (52 three-of-a-kinds + 48 suited runs).
   - It splits its 6 cards into two groups of 3 (only 10 ways!) and counts how many **live unseen cards** can finish each group—subtracting dead cards already buried in the discard pile.
   - It also watches what the next player picked from `Shown` so it avoids feeding them helpful cards.

4. **1-Level Tree Bot — Medium ([bots/tree_bot.py](bots/tree_bot.py)):**
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

5. **2-Level Tree Bot — Hard ([bots/tree_bot.py](bots/tree_bot.py)):**
   - Takes the top candidate moves from Level 1 and looks a **second turn ahead** (`Draw 1 → Discard 1 → Draw 2 → Discard 2`), discounting future turns slightly (`gamma = 0.85`) so winning sooner is always preferred.

### Pit Bots Against Each Other (No Browser Needed!)
```bash
python3 -m scripts.evaluate --bot-a heuristic --bot-b random --games 60
python3 -m scripts.evaluate --bot-a tree_1ply --bot-b heuristic --games 60
python3 -m scripts.evaluate --bot-a rl_q_learning --bot-b heuristic --games 60
```
You can even generate bot self-play training data with `--save-trajectories selfplay.jsonl`!

---

## Step 3: The 4 PyTorch Reinforcement Learning (RL) Players (`ml/`)

All four RL bots are written in **clean, idiomatic PyTorch** (`nn.Module`, `nn.Sequential`, `optim.Adam`, `loss.backward()`, `torch.distributions.Categorical`, safe `weights_only=True` `.pt` checkpoints) sharing 24 suit-invariant features (including **opponent memory & defense** features like `opp_danger_exact` and `safe_discard_score`) in [ml/features.py](ml/features.py):

1. **🧠 RL Bot — TD Value ([ml/td_value.py](ml/td_value.py)):**
   - Learns a **state value function** $V(s) \in (0, 1)$ predicting win probability from any 6-card hand using **live target-network TD($\lambda$)** + `F.binary_cross_entropy` + Boltzmann exploration (just like TD-Gammon!).
2. **🧠 RL Bot — Double DQN ([ml/q_learning.py](ml/q_learning.py)):**
   - Learns explicit **action values** $Q(s, a) \in (-1, 1)$ for every legal `DRAW` and `DISCARD` using **Double DQN** (online argmax, slow `tau=0.005` target net evaluation), MC-blended credit assignment, and `F.smooth_l1_loss`. Needs zero draw simulation (~0.2 ms/move)!
3. **🧠 RL Bot — PPO Policy Gradient ([ml/policy_gradient.py](ml/policy_gradient.py)):**
   - **Batched Proximal Policy Optimization (PPO):** collects 64-game rollout batches, computes normalized **GAE** advantages, and runs clipped policy-ratio epochs over masked `torch.distributions.Categorical` distributions.
4. **🧠 RL + Tree Bot — AlphaZero-Lite ([ml/alphazero_bot.py](ml/alphazero_bot.py)):**
   - Combines a **Policy Prior network**, a **Value network**, and **defense-preserving 1-Level Expectimax** — training itself directly from self-play search targets (`models/alphazero_model.pt`)!

### Train All 3 PyTorch RL Bots Yourself (10,000 Progressive Elo Games)
```bash
pip install -r requirements-ml.txt
python3 -m scripts.train_rl --games 10000 --save-trajectories selfplay_10k.jsonl.gz
```
The curriculum automatically gets harder across 4 stages (Starter/Beginner → Heuristic → Top-Elo Pool → Champions League), finishes in ~3 minutes on CPU, saves compact gzip-compressed trajectories to gitignored `selfplay_10k.jsonl.gz` (~14 MB for 10,000 games!), saves PyTorch checkpoints to `models/*.pt`, and runs a round-robin tournament across all 8 players:

| Rank | Player | Tournament Elo | Overall Win% |
|---:|---|---:|---:|
| 1 | Heuristic Bot (Easy) | 1144 | 66.0% |
| 2 | 🧠 RL Bot — Policy Gradient (PyTorch) | 1142 | **67.9%** |
| 3 | Tree Bot (2-Level) | 1130 | 63.1% |
| 4 | 🧠 RL Bot — Q-Learning (PyTorch) | 1121 | 60.3% |
| 5 | Tree Bot (1-Level) | 1101 | 66.0% |
| 6 | 🧠 RL Bot — TD Value (PyTorch) | 1078 | 58.3% |
| 7 | Beginner Bot (Very Easy) | 708 | 9.6% |
| 8 | Random Bot (Starter) | 576 | 0.0% |

All three trained PyTorch checkpoints are included in `models/` and appear right in the lobby dropdown once PyTorch is installed!

---

## Step 4: Generational Bot League & Interactive Bot Lab (`scripts/league.py`)

Want to watch all four PyTorch RL bots **play against each other in continuous 5,000-game batches** and track which algorithm improves fastest? Run the Generational League:

```bash
# Train 4 batches x 5,000 games = 20,000 games (paired duplicate-deal eval!)
python3 -m scripts.league --generations 4 --games-per-gen 5000 --include-human trajectories.jsonl

# Keep training continuously in more 5,000-game batches from your saved checkpoints!
python3 -m scripts.league --resume --generations 4 --games-per-gen 5000 --include-human trajectories.jsonl
```

Each 5,000-game batch plays RL-vs-RL, frozen Hall-of-Fame champions, Heuristic, Tree 1-Level, and Novice anchor games; evaluates all 9 players using **Paired Duplicate Deals** (same shuffled deck from both seats!) + **order-independent Bradley-Terry Elo**; updates `models/league_history.json` live after every batch; and uses **King-of-the-Hill gating** so checkpoints in `models/*.pt` only advance when Elo improves!

- **Interactive Bot Lab (`/league`):** Click **`📊 Bot League`** in the lobby (or visit `http://localhost:7171/league`) to explore the interactive SVG Elo chart (click legend items to toggle, hover for exact stats), run instant **Quick Bot-vs-Bot Matchups**, and test any bot's thinking live in the **Bot Brain Sandbox**!
- **Watch Bots Battle Live (Spectator Arena):** Add any 2–4 computer players in the lobby without joining as a human and click **Start Game**! You can adjust **Bot Speed** (`Slow` / `Normal` / `Fast`), see every bot's cards live, and read the **🧠 Bot Brain** box explaining why each bot chose its move (fairness check: Bot Brain & open bot hands are automatically hidden whenever a human is playing!).

### 8-Batch League Progression (40,000 Games, Paired Duplicate Deals)
| Batch | Games | 🧠 TD Value | 🧠 Double DQN | 🧠 PPO | 🧠 AlphaZero | Heuristic | Tree 1-Ply |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gen 0 (Untrained) | 0 | 1169 | 730 | 831 | 762 | 1335 | 1338 |
| Gen 1 | 5,000 | 1139 | 1132 | 1082 | 1106 | 1142 | 1142 |
| Gen 2 | 10,000 | 1132 | 1112 | 1107 | 1077 | 1152 | 1190 |
| Gen 3 | 15,000 | **1163** | 1108 | 1115 | 1088 | 1133 | 1181 |
| Gen 4 | 20,000 | **1163** | 1116 | **1129** | 1084 | 1160 | 1173 |
| Gen 5 *(resume)* | 25,000 | **1162 (#1!)** | 1118 | 1090 | 1099 | 1138 | 1148 |
| Gen 6 *(resume)* | 30,000 | 1124 | 1137 | 1087 | 1070 | 1134 | 1150 |
| Gen 7 *(resume)* | 35,000 | 1116 | **1151** | 1102 | 1100 | 1132 | 1159 |
| **Gen 8 (Final)** | **40,000** | **1126** | **1146 (T-#1, 65.1%!)** | **1110** | **1096 (55% vs T1!)** | **1136** | **1146** |

---

## Step 5: Beating Card-Luck Variance (`scripts/train_variance_reduced.py`)

Because Rummy has high card-deal luck, single `+1 / -1` game outcomes are noisy. Two modern RL tricks slash that variance cleanly (without ever cheating at play time!):

1. **🧠 RL Bot — X-Ray GRPO ([ml/grpo_bot.py](ml/grpo_bot.py)):**
   - **Omniscient ("X-Ray") Critic (training only):** Like Microsoft's Suphx Mahjong AI, the Critic sees *both* players' real hands during training so it doesn't blame the Actor for an unwinnable opponent deal, while the Actor remains 100% fair at play time.
   - **Duplicate-Deal GRPO:** Every training deck is played twice with seats swapped (mirror deals), cancelling out pure deal luck.
2. **🧠 RL Bot — Rollout Distilled ([ml/distill_bot.py](ml/distill_bot.py)):**
   - **Expert Iteration (LLM-style Distillation):** Freezes real game turns (strictly legal info only), runs 12 Common-Random-Number valid unseen worlds per legal move to measure true EVs, and distills the Teacher's sharp probabilities into a fast ~0.2 ms PyTorch Student — then plugs the Student back in as the rollout player for Iteration 2!

```bash
python3 -m scripts.train_variance_reduced --grpo-batches 4 --games-per-batch 5000 --distill-iters 2
```

### 5,000-Game Luck-Adjusted Duplicate-Deal Benchmark (2,500g vs Heuristic + 2,500g vs Tree 1-Level)
Measuring **excess skill over card luck** (stripping out `1–1` luck splits and counting `2–0` vs `0–2` mirror sweeps across 2,500 paired duplicate deals per bot, `±0.57%` SE):

| Rank | Bot / Ensemble | 5k Score (`±SE`) | vs Heuristic | vs Tree 1-Level | Skill Sweeps (`2–0`/`0–2`) | Decisive Win% | Excess / 100 Deals | McNemar `z` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | 🧠 Double DQN (PyTorch) | **50.77%** (±0.56%) | **50.90%** | **50.64%** | **405 – 367** | **52.5%** | **+1.52** | **+1.37** |
| **2** | 🧠 TD Value (PyTorch) | **50.67%** (±0.57%) | **51.20%** | **50.14%** | **426 – 394** | **52.0%** | **+1.28** | **+1.12** |
| **3** | Ensemble: Top-4 RL (Soft Prob) | **50.56%** (±0.57%) | **51.14%** | 49.98% | 415 – 387 | 51.7% | +1.12 | +0.99 |
| **4** | Ensemble: Top-5 RL (Mode Vote) | **50.53%** (±0.57%) | **50.72%** | **50.34%** | 424 – 397 | 51.6% | +1.08 | +0.94 |
| **5** | 🧠 X-Ray GRPO (PyTorch) | **50.43%** (±0.58%) | **50.88%** | 49.98% | 429 – 406 | 51.4% | +0.92 | +0.80 |
| **6** | 🧠 PPO Policy Gradient | 49.90% (±0.57%) | 50.52% | 49.28% | 402 – 407 | 49.7% | -0.20 | -0.18 |
| **7** | 🧠 Rollout Distilled *(900 states)* | 49.51% (±0.59%) | 50.00% | 49.02% | 424 – 449 | 48.6% | -1.00 | -0.85 |

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
- [x] **Step 3 — ML / RL Computer Players:** Trained three idiomatic PyTorch RL bots (TD Value, Q-Learning, and A2C Policy Gradient) via 10,000 progressive Elo self-play games and plugged all three directly into the lobby dropdown and leaderboard!
