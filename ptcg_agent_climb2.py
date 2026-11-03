"""Pokemon TCG AI Battle Challenge agent -- single-file submission.

Determinized flat Monte-Carlo search over the engine's forward model, with a rule-based
heuristic as rollout policy and fallback. Harness/agent code adapted from the MIT-licensed
public repository by Kelly Mao (github.com/Makkkelyie/pokemon-tcg-ai); deck tuned locally.
The last `def` in this file is the Kaggle agent: agent(obs) -> list[int].
"""
import ctypes
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

# --------------------------------------------------------------------------- engine
def _load_cg():
    import importlib
    try:
        cg = importlib.import_module("cg")  # shipped next to the agent, or on sys.path
        importlib.import_module("cg.sim"); importlib.import_module("cg.game")
    except ImportError:
        cg = importlib.import_module("kaggle_environments.envs.cabt.cg")  # bundled with the environment
        importlib.import_module("kaggle_environments.envs.cabt.cg.sim")
        importlib.import_module("kaggle_environments.envs.cabt.cg.game")
    lib = cg.sim.lib
    P = ctypes.POINTER(ctypes.c_int)
    try:
        lib.AgentStart.restype = ctypes.c_void_p; lib.AgentStart.argtypes = []
        lib.SearchBegin.restype = ctypes.c_char_p
        lib.SearchBegin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, P, P, P, P, P, P, ctypes.c_int]
        lib.SearchStep.restype = ctypes.c_char_p
        lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int, P, ctypes.c_int]
        lib.SearchRelease.restype = None; lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.SearchEnd.restype = None; lib.SearchEnd.argtypes = [ctypes.c_void_p]
    except AttributeError:
        pass
    return cg


_CG = _load_cg()


def ensure_cg():
    return _CG


def _ns(d):
    return SimpleNamespace(**{k: ([_ns(x) if isinstance(x, dict) else x for x in v] if isinstance(v, list)
                               else (_ns(v) if isinstance(v, dict) else v)) for k, v in d.items()})


def _all_card_data():
    try:
        return _CG.api.all_card_data()
    except Exception:
        lib = _CG.sim.lib
        lib.AllCard.restype = ctypes.c_char_p; lib.AllCard.argtypes = []
        return [_ns(c) for c in json.loads(lib.AllCard().decode())]


def _all_attack():
    try:
        return _CG.api.all_attack()
    except Exception:
        lib = _CG.sim.lib
        lib.AllAttack.restype = ctypes.c_char_p; lib.AllAttack.argtypes = []
        return [_ns(a) for a in json.loads(lib.AllAttack().decode())]


# ======================================================================
# ptcg/cards.py
# ======================================================================
"""Card / attack metadata from the engine, as plain dicts keyed by id."""


# Enum values from cg/api.py (kept as ints for speed).
POKEMON, ITEM, TOOL, SUPPORTER, STADIUM, BASIC_ENERGY, SPECIAL_ENERGY = range(7)
COLORLESS = 0


@dataclass
class AttackInfo:
    id: int
    name: str
    damage: int
    cost: list[int]
    text: str


@dataclass
class CardInfo:
    id: int
    name: str
    type: int
    hp: int
    retreat: int
    weakness: int | None
    resistance: int | None
    energy_type: int
    basic: bool
    stage: int  # 0 basic, 1, 2 (for Pokémon)
    prize_value: int  # prizes the opponent takes on KO
    tera: bool
    evolves_from: str | None
    attacks: list[AttackInfo] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)

    @property
    def is_pokemon(self) -> bool:
        return self.type == POKEMON

    @property
    def is_energy(self) -> bool:
        return self.type in (BASIC_ENERGY, SPECIAL_ENERGY)

    @property
    def max_damage(self) -> int:
        return max((a.damage for a in self.attacks), default=0)


class CardDB:
    def __init__(self):
        self.attacks = {
            a.attackId: AttackInfo(a.attackId, a.name, a.damage, list(a.energies), a.text)
            for a in _all_attack()
        }
        self.cards: dict[int, CardInfo] = {}
        for c in _all_card_data():
            stage = 2 if c.stage2 else 1 if c.stage1 else 0
            prize = 3 if c.megaEx else 2 if c.ex else 1
            self.cards[c.cardId] = CardInfo(
                id=c.cardId,
                name=c.name,
                type=int(c.cardType),
                hp=c.hp,
                retreat=c.retreatCost,
                weakness=c.weakness,
                resistance=c.resistance,
                energy_type=int(c.energyType),
                basic=c.basic,
                stage=stage,
                prize_value=prize,
                tera=c.tera,
                evolves_from=c.evolvesFrom,
                attacks=[self.attacks[i] for i in c.attacks if i in self.attacks],
                skills=[s.text for s in c.skills],
            )
        self.by_name: dict[str, list[int]] = {}
        for c in self.cards.values():
            self.by_name.setdefault(c.name, []).append(c.id)

    def __getitem__(self, cid: int) -> CardInfo:
        return self.cards[cid]

    def get(self, cid):
        return self.cards.get(cid)

    def evolves_into(self, base_id: int, evo_id: int) -> bool:
        evo = self.cards.get(evo_id)
        base = self.cards.get(base_id)
        return bool(evo and base and evo.evolves_from == base.name)


@lru_cache(maxsize=1)
def card_db() -> CardDB:
    return CardDB()


def energy_satisfies(attached: list[int], cost: list[int]) -> bool:
    """True if attached energy units pay `cost` (colorless = any; RAINBOW pays any type)."""
    pool = list(attached)
    need_colorless = 0
    for e in cost:
        if e == COLORLESS:
            need_colorless += 1
            continue
        if e in pool:
            pool.remove(e)
        elif 10 in pool:  # RAINBOW
            pool.remove(10)
        elif 11 in pool and e in (5, 7):  # TEAM_ROCKET: psychic or darkness
            pool.remove(11)
        else:
            return False
    return len(pool) >= need_colorless


def missing_energy(attached: list[int], cost: list[int]) -> int:
    """Rough number of extra energy units needed to pay `cost`."""
    if energy_satisfies(attached, cost):
        return 0
    return max(1, len(cost) - len(attached))

# ======================================================================
# ptcg/state.py
# ======================================================================
"""Helpers for reading observation dicts (the raw JSON the engine produces)."""

# AreaType
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM, ENERGY_AREA, TOOL_AREA, PRE_EVO, PLAYER, LOOKING = range(1, 13)

# OptionType
O_NUMBER, O_YES, O_NO, O_CARD, O_TOOL_CARD, O_ENERGY_CARD, O_ENERGY, O_PLAY, O_ATTACH, O_EVOLVE, \
    O_ABILITY, O_DISCARD, O_RETREAT, O_ATTACK, O_END, O_SKILL, O_SPECIAL = range(17)

# SelectType
S_MAIN, S_CARD, S_ATTACHED, S_CARD_OR_ATTACHED, S_ENERGY, S_SKILL, S_ATTACK, S_EVOLVE, S_COUNT, \
    S_YES_NO, S_SPECIAL = range(11)


def me(obs) -> int:
    return obs["current"]["yourIndex"]


def players(obs):
    cur = obs["current"]
    i = cur["yourIndex"]
    return cur["players"][i], cur["players"][1 - i]


def in_play(ps) -> list[dict]:
    """Active (if face-up) followed by bench Pokémon."""
    return [p for p in ps["active"] if p] + list(ps["bench"])


def card_at(obs, area, index, player):
    """Resolve an (area, index, player) reference to a card/Pokémon dict, or None."""
    try:
        cur = obs["current"]
        ps = cur["players"][player]
        if area == DECK:
            return obs["select"]["deck"][index]
        if area == HAND:
            h = ps["hand"]
            return h[index] if h else None
        if area == DISCARD:
            return ps["discard"][index]
        if area == ACTIVE:
            return ps["active"][index]
        if area == BENCH:
            return ps["bench"][index]
        if area == PRIZE:
            return ps["prize"][index]
        if area == STADIUM:
            return cur["stadium"][index]
        if area == LOOKING:
            return cur["looking"][index]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def prizes_left(ps) -> int:
    return len(ps["prize"])


def best_ready_damage(poke) -> int:
    """Highest printed damage among attacks this Pokémon can currently pay for."""
    info = card_db().get(poke["id"])
    if not info:
        return 0
    best = 0
    for a in info.attacks:
        if energy_satisfies(poke["energies"], a.cost):
            best = max(best, attack_value(a))
    return best


def attack_value(a) -> int:
    """Printed damage, with a rough guess for 0-damage attacks whose text deals damage."""
    if a.damage > 0:
        return a.damage
    t = a.text.lower()
    if "damage" in t and "this attack does" in t:
        return 60
    return 10


def energy_need(poke) -> int:
    """Energy units still missing for this Pokémon's strongest attack."""
    info = card_db().get(poke["id"])
    if not info or not info.attacks:
        return 0
    best = max(info.attacks, key=attack_value)
    return missing_energy(poke["energies"], best.cost)


def effective_damage(attacker_id, dmg, defender) -> int:
    db = card_db()
    a, d = db.get(attacker_id), db.get(defender["id"]) if defender else None
    if not a or not d or dmg <= 0:
        return dmg
    if d.weakness is not None and d.weakness == a.energy_type and a.energy_type != 0:
        dmg *= 2
    if d.resistance is not None and d.resistance == a.energy_type and a.energy_type != 0:
        dmg = max(0, dmg - 30)
    return dmg


def prize_value(poke) -> int:
    info = card_db().get(poke["id"]) if poke else None
    return info.prize_value if info else 1

# ======================================================================
# ptcg/evaluate.py
# ======================================================================
"""Static evaluation of a (possibly simulated) game state, in [-1, 1] for `me`."""


DEFAULT_WEIGHTS = {
    "prize": 1.0,         # per prize card taken (difference)
    "ko_progress": 0.35,  # damage fraction on opponent Pokémon x prize value
    "board": 0.12,        # per Pokémon in play
    "energy": 0.08,       # per useful attached energy unit
    "stage": 0.10,        # per evolution stage in play
    "ready": 0.25,        # active can attack now
    "can_ko": 0.30,       # active can KO opponent's active (x prize value)
    "threat": 0.30,       # opponent's active can KO ours (x prize value)
    "hand": 0.03,         # per card in hand (capped)
    "no_bench": 0.6,      # penalty: no benched Pokémon
    "deck_low": 0.15,     # per card below 5 left in deck
}


def _side(ps, w):
    s = 0.0
    db = card_db()
    pokes = in_play(ps)
    for p in pokes:
        info = db.get(p["id"])
        s += w["board"]
        if info:
            s += w["stage"] * info.stage
            best = max((len(a.cost) for a in info.attacks), default=0)
            s += w["energy"] * min(len(p["energies"]), best)
    s += w["hand"] * min(ps["handCount"], 8)
    if not ps["bench"]:
        s -= w["no_bench"]
    if ps["deckCount"] < 5:
        s -= w["deck_low"] * (5 - ps["deckCount"])
    return s


def _ko_progress(ps):
    t = 0.0
    for p in in_play(ps):
        if p.get("maxHp"):
            t += (1 - p["hp"] / p["maxHp"]) * prize_value(p)
    return t


def _threat(att, dfd, extra_energy=1):
    """Could `att` (with one more energy attached) KO `dfd` with a printed attack?"""
    if att is None or dfd is None:
        return False
    info = card_db().get(att["id"])
    if not info:
        return False
    n = len(att["energies"]) + extra_energy
    for a in info.attacks:
        if len(a.cost) <= n and effective_damage(att["id"], attack_value(a), dfd) >= dfd["hp"]:
            return True
    return False


def evaluate(obs, me: int, w=DEFAULT_WEIGHTS) -> float:
    cur = obs["current"]
    if cur["result"] >= 0:
        return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
    P, O = cur["players"][me], cur["players"][1 - me]
    s = w["prize"] * ((6 - len(P["prize"])) - (6 - len(O["prize"])))
    s += w["ko_progress"] * (_ko_progress(O) - _ko_progress(P))
    s += _side(P, w) - _side(O, w)
    my_act = P["active"][0] if P["active"] else None
    op_act = O["active"][0] if O["active"] else None
    if my_act:
        dmg = best_ready_damage(my_act)
        if dmg:
            s += w["ready"]
            if op_act and effective_damage(my_act["id"], dmg, op_act) >= op_act["hp"]:
                s += w["can_ko"] * prize_value(op_act)
        if _threat(op_act, my_act):
            s -= w["threat"] * prize_value(my_act)
    if op_act:
        if best_ready_damage(op_act):
            s -= w["ready"]
    return math.tanh(s / 3.0)

# ======================================================================
# ptcg/search_api.py
# ======================================================================
"""Thin dict-based wrapper over the engine's search functions (forward model).

cg.api.search_begin/step build nested dataclasses on every call; for MCTS we
only need the raw JSON, so this calls the C library directly and returns dicts
shaped exactly like the observations the agent receives.
"""


_agent_ptr = None


def _lib():
    return ensure_cg().sim.lib


def _ptr():
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = _lib().AgentStart()
    return _agent_ptr


class SearchError(Exception):
    pass


def _arr(xs):
    return (ctypes.c_int * len(xs))(*xs)


def begin(obs: dict, my_deck, my_prize, op_deck, op_prize, op_hand, op_active, manual_coin=False):
    """Start a determinized search from `obs`. Returns (search_id, observation dict)."""
    sbi = obs.get("search_begin_input")
    if not sbi:
        raise SearchError("observation has no search_begin_input")
    if obs["select"].get("deck") is not None:
        my_deck = []
    bs = _lib().SearchBegin(
        _ptr(), sbi.encode("ascii"), len(sbi),
        _arr(my_deck), _arr(my_prize), _arr(op_deck), _arr(op_prize),
        _arr(op_hand), _arr(op_active), int(manual_coin),
    )
    res = json.loads(bs.decode())
    if res["error"] != 0:
        raise SearchError(f"SearchBegin error {res['error']}")
    st = res["state"]
    return st["searchId"], st["observation"]


def step(search_id: int, select: list[int]):
    bs = _lib().SearchStep(_ptr(), search_id, _arr(select), len(select))
    res = json.loads(bs.decode())
    if res["error"] != 0:
        raise SearchError(f"SearchStep error {res['error']}")
    st = res["state"]
    return st["searchId"], st["observation"]


def release(search_id: int):
    _lib().SearchRelease(_ptr(), search_id)


def end():
    """Free all search states (call once per decision)."""
    _lib().SearchEnd(_ptr())

# ======================================================================
# ptcg/belief.py
# ======================================================================
"""Hidden-information tracking and determinization for the search API.

search_begin needs concrete card ids for every hidden zone (our deck order and
prizes, the opponent's deck/prizes/hand and a face-down Active). We know our
own 60-card list exactly; for the opponent we keep, per card id, the largest
number of copies ever visible at once (board, attachments, discard, stadium)
over the whole BO3 episode -- the same deck is used in all games -- and fill
the rest with basic Energy of their most common type.
"""



def _visible_ids(ps, include_hand=True) -> Counter:
    c = Counter()
    for p in in_play(ps):
        c[p["id"]] += 1
        for e in p.get("energyCards", []):
            c[e["id"]] += 1
        for t in p.get("tools", []):
            c[t["id"]] += 1
        for pe in p.get("preEvolution", []):
            c[pe["id"]] += 1
    for d in ps["discard"]:
        c[d["id"]] += 1
    for pr in ps["prize"]:
        if pr:
            c[pr["id"]] += 1
    if include_hand and ps.get("hand"):
        for h in ps["hand"]:
            c[h["id"]] += 1
    return c


class Belief:
    def __init__(self, my_deck: list[int]):
        self.my_deck = Counter(my_deck)
        self.my_filler = self.my_deck.most_common(1)[0][0]
        self.op_seen = Counter()
        db = card_db()
        self.basic_energy_of_type = {}
        for c in db.cards.values():
            if c.type == BASIC_ENERGY and c.energy_type not in self.basic_energy_of_type:
                self.basic_energy_of_type[c.energy_type] = c.id
        self.my_basic = next((cid for cid in my_deck if db[cid].is_pokemon and db[cid].basic), None)

    def observe(self, obs):
        cur = obs.get("current")
        if not cur:
            return
        i = cur["yourIndex"]
        op = cur["players"][1 - i]
        vis = _visible_ids(op, include_hand=False)
        for st in cur.get("stadium") or []:
            if st["playerIndex"] == 1 - i:
                vis[st["id"]] += 1
        for cid, n in vis.items():
            if n > self.op_seen[cid]:
                self.op_seen[cid] = n
        for log in obs.get("logs") or []:
            # Cards the opponent played/attached/evolved are revealed even if they leave play.
            if log.get("playerIndex") == 1 - i and log.get("type") in (10, 11, 12) and log.get("cardId"):
                cid = log["cardId"]
                if self.op_seen[cid] == 0:
                    self.op_seen[cid] = 1

    # ------------------------------------------------------------------
    def _op_filler(self) -> int:
        db = card_db()
        counts = Counter()
        for cid, n in self.op_seen.items():
            info = db.get(cid)
            if info and info.type == BASIC_ENERGY:
                counts[cid] += n
        if counts:
            return counts.most_common(1)[0][0]
        types = Counter()
        for cid, n in self.op_seen.items():
            info = db.get(cid)
            if info and info.is_pokemon:
                for a in info.attacks:
                    for e in a.cost:
                        if e:
                            types[e] += n
        if types:
            t = types.most_common(1)[0][0]
            if t in self.basic_energy_of_type:
                return self.basic_energy_of_type[t]
        return self.my_filler

    def determinize(self, obs, rng):
        """Return kwargs for begin with a random consistent assignment."""
        cur = obs["current"]
        i = cur["yourIndex"]
        me, op = cur["players"][i], cur["players"][1 - i]

        # --- our hidden cards: deck list minus everything we can see
        vis = _visible_ids(me)
        for st in cur.get("stadium") or []:
            if st["playerIndex"] == i:
                vis[st["id"]] += 1
        for lk in cur.get("looking") or []:
            if lk and lk.get("playerIndex", i) == i:
                vis[lk["id"]] += 1
        sel_deck = obs["select"].get("deck")
        if sel_deck:
            for c in sel_deck:
                vis[c["id"]] += 1
        pool = []
        for cid, n in self.my_deck.items():
            pool += [cid] * max(0, n - vis[cid])
        rng.shuffle(pool)
        n_fd_prize = sum(1 for p in me["prize"] if p is None)
        need = (0 if sel_deck else me["deckCount"]) + n_fd_prize
        pool = (pool + [self.my_filler] * need)[:need]
        my_prize_fd = pool[:n_fd_prize]
        my_deck = pool[n_fd_prize:]
        if sel_deck:
            my_deck = []
        my_prize = []
        it = iter(my_prize_fd)
        for p in me["prize"]:
            my_prize.append(p["id"] if p else next(it))

        # --- opponent's hidden cards
        ovis = _visible_ids(op, include_hand=False)
        hidden = []
        for cid, n in self.op_seen.items():
            hidden += [cid] * max(0, n - ovis[cid])
        rng.shuffle(hidden)
        n_op_fd_prize = sum(1 for p in op["prize"] if p is None)
        face_down_active = len(op["active"]) > 0 and op["active"][0] is None
        n_hidden = op["deckCount"] + op["handCount"] + n_op_fd_prize
        filler = self._op_filler()
        hidden = (hidden[:n_hidden] + [filler] * n_hidden)[:n_hidden]
        rng.shuffle(hidden)
        op_hand = hidden[: op["handCount"]]
        rest = hidden[op["handCount"]:]
        op_prize_fd = rest[:n_op_fd_prize]
        op_deck = rest[n_op_fd_prize:]
        op_prize = []
        it = iter(op_prize_fd)
        for p in op["prize"]:
            op_prize.append(p["id"] if p else next(it))
        op_active = []
        if face_down_active:
            db = card_db()
            basics = [c for c in self.op_seen if db[c].is_pokemon and db[c].basic]
            op_active = [rng.choice(basics) if basics else self.my_basic]
        if cur["turn"] == 0 and self.my_basic is not None:
            # Setup needs at least one Basic Pokémon in the opponent's deck.
            db = card_db()
            if not any(db[c].is_pokemon and db[c].basic for c in op_deck) and op_deck:
                op_deck[0] = self.my_basic
        return dict(my_deck=my_deck, my_prize=my_prize, op_deck=op_deck, op_prize=op_prize,
                    op_hand=op_hand, op_active=op_active)

# ======================================================================
# ptcg/agents/base.py
# ======================================================================
"""Agent base class.

Kaggle calls `agent(obs_dict)` once per decision. When obs["select"] is None the
agent must return its 60-card deck; otherwise a list of option indices with
minCount <= len <= maxCount and no duplicates.
"""


class Agent:
    name = "base"

    def __init__(self, deck: list[int], seed: int | None = None):
        if len(deck) != 60:
            raise ValueError("deck must have 60 cards")
        self.deck = list(deck)
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            self.new_match()
            return list(self.deck)
        return self.act(obs)

    def new_match(self):
        """Hook: called when a new BO3 episode starts (deck request)."""

    def act(self, obs: dict) -> list[int]:
        raise NotImplementedError


def random_choice(select: dict, rng: random.Random) -> list[int]:
    n = len(select["option"])
    k = rng.randint(select["minCount"], select["maxCount"])
    return rng.sample(range(n), k)


class RandomAgent(Agent):
    """Uniformly random legal action (number of picks also random)."""

    name = "random"

    def act(self, obs):
        return random_choice(obs["select"], self.rng)


class FirstAgent(Agent):
    """Always picks the first maxCount options (the env's 'first' builtin)."""

    name = "first"

    def act(self, obs):
        return list(range(obs["select"]["maxCount"]))

# ======================================================================
# ptcg/agents/heuristic.py
# ======================================================================
"""Rule-based agent: scores every legal option, then picks the best ones.

Scores are designed so that options we *want* are > 0 and options we only take
when forced are <= 0; `pick` then chooses between minCount and maxCount of
them. The same scorer is reused as the rollout policy inside the search agent.
"""


# SelectContext values (cg/api.py)
C_MAIN, C_SETUP_ACTIVE, C_SETUP_BENCH, C_SWITCH, C_TO_ACTIVE, C_TO_BENCH, C_TO_FIELD, C_TO_HAND, \
    C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE, C_NOT_MOVE, C_DMG_COUNTER, C_DMG_COUNTER_ANY, \
    C_DAMAGE, C_REMOVE_DMG, C_HEAL, C_EVOLVES_FROM, C_EVOLVES_TO, C_DEVOLVE, C_ATTACH_FROM, C_ATTACH_TO, \
    C_DETACH_FROM, C_LOOK, C_EFFECT_TARGET, C_DISCARD_ENERGY_CARD, C_DISCARD_TOOL, C_SWITCH_ENERGY_CARD, \
    C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY, C_TO_HAND_ENERGY, C_TO_DECK_ENERGY, C_SWITCH_ENERGY, \
    C_SKILL_ORDER, C_ATTACK, C_DISABLE_ATTACK, C_EVOLVE, C_DRAW_COUNT, C_DMG_COUNTER_COUNT, \
    C_REMOVE_DMG_COUNT, C_IS_FIRST, C_MULLIGAN, C_ACTIVATE, C_FIRST_EFFECT, C_MORE_DEVOLVE, C_COIN_HEAD, \
    C_AFFECT_SPECIAL, C_RECOVER_SPECIAL = range(49)

# Contexts where moving a card away from us is a cost (prefer minCount, least valuable).
LOSS_CONTEXTS = {C_DISCARD, C_TO_DECK, C_TO_DECK_BOTTOM, C_TO_PRIZE, C_DISCARD_ENERGY_CARD, C_DISCARD_TOOL,
                 C_DISCARD_CARD_OR_ATTACHED, C_DISCARD_ENERGY, C_TO_HAND_ENERGY, C_TO_DECK_ENERGY, C_NOT_MOVE}
# Contexts where we gain the selected cards (prefer maxCount, most valuable).
GAIN_CONTEXTS = {C_TO_HAND, C_TO_BENCH, C_TO_FIELD, C_SETUP_BENCH, C_EVOLVES_TO, C_LOOK}
CONDITION_PRIORITY = {3: 5, 2: 4, 0: 3, 1: 2, 4: 1}  # paralyze > sleep > poison > burn > confuse


class Ctx:
    """Per-decision cached view of the observation."""

    def __init__(self, obs):
        self.obs = obs
        self.db = card_db()
        cur = obs["current"]
        self.cur = cur
        self.i = cur["yourIndex"]
        self.P = cur["players"][self.i]
        self.O = cur["players"][1 - self.i]
        self.select = obs["select"]
        self.hand = self.P["hand"] or []
        self.my_pokes = in_play(self.P)
        self.op_active = self.O["active"][0] if self.O["active"] else None
        self.my_active = self.P["active"][0] if self.P["active"] else None
        names = set()
        for p in self.my_pokes:
            info = self.db.get(p["id"])
            if info:
                names.add(info.name)
        self.my_names = names
        hand_evo_from = set()
        for c in self.hand:
            info = self.db.get(c["id"])
            if info and info.is_pokemon and info.evolves_from:
                hand_evo_from.add(info.evolves_from)
        self.hand_evo_from = hand_evo_from

    def info(self, cid):
        return self.db.get(cid)

    def poke_ref(self, area, index, player):
        return card_at(self.obs, area, index, player)


# ------------------------------------------------------------------ values

def card_value(ctx: Ctx, cid: int) -> float:
    """How much we'd like to have this card in hand right now."""
    info = ctx.info(cid)
    if info is None:
        return 10.0
    t = info.type
    if t == POKEMON:
        if info.stage > 0:
            return 70.0 if info.evolves_from in ctx.my_names else 35.0
        bench_room = ctx.P["benchMax"] - len(ctx.P["bench"])
        base = 60.0 if len(ctx.my_pokes) < 3 else 35.0
        if info.name in ctx.hand_evo_from:
            base += 15
        return base if bench_room > 0 else 15.0
    if t in (BASIC_ENERGY, SPECIAL_ENERGY):
        needs = sum(1 for p in ctx.my_pokes if energy_need(p) > 0)
        return 45.0 if needs else 20.0
    if t == SUPPORTER:
        return 45.0
    if t == ITEM:
        return 38.0
    if t == TOOL:
        return 28.0
    if t == STADIUM:
        return 18.0
    return 20.0


def poke_readiness(ctx: Ctx, poke) -> float:
    """How good this Pokémon would be in the Active Spot."""
    if poke is None:
        return 0.0
    dmg = best_ready_damage(poke)
    need = energy_need(poke)
    hp = poke.get("hp", 0)
    score = dmg + 0.3 * hp - 25 * need
    if ctx.op_active is not None and dmg:
        eff = effective_damage(poke["id"], dmg, ctx.op_active)
        if eff >= ctx.op_active["hp"]:
            score += 150
    return score


def target_value_for_damage(ctx: Ctx, poke, dmg=30) -> float:
    """Value of hitting opponent Pokémon `poke` with ~dmg damage."""
    if poke is None:
        return 0.0
    hp = poke["hp"]
    v = 100.0 - hp / 5.0
    if hp <= dmg:
        v += 120 * prize_value(poke)
    return v


# --------------------------------------------------------------- MAIN phase

def score_main(ctx: Ctx, opt) -> float:
    t = opt["type"]
    if t == O_END:
        return 0.0
    if t == O_ATTACK:
        a = ctx.db.attacks.get(opt.get("attackId"))
        if a is None or ctx.my_active is None:
            return 300.0
        dmg = effective_damage(ctx.my_active["id"], attack_value(a), ctx.op_active)
        s = 300.0 + dmg / 5.0
        if ctx.op_active is not None and dmg >= ctx.op_active["hp"]:
            s += 100 + 20 * prize_value(ctx.op_active)
        return s
    if t == O_EVOLVE:
        info = ctx.info(_hand_id(ctx, opt))
        return 860.0 + (10 * info.stage if info else 0)
    if t == O_PLAY:
        cid = _hand_id(ctx, opt)
        info = ctx.info(cid)
        if info is None:
            return 500.0
        if info.type == POKEMON:
            room = ctx.P["benchMax"] - len(ctx.P["bench"])
            if room <= 0:
                return -10.0
            s = 820.0 - 25 * len(ctx.P["bench"])
            if info.name in ctx.hand_evo_from:
                s += 20
            return s
        if info.type == ITEM:
            return 770.0
        if info.type == SUPPORTER:
            return 740.0
        if info.type == TOOL:
            return 650.0
        if info.type == STADIUM:
            cur_st = ctx.cur["stadium"]
            return 600.0 if not cur_st or cur_st[0]["playerIndex"] != ctx.i else -5.0
        return 500.0
    if t == O_ATTACH:
        cid = _hand_id(ctx, opt)
        info = ctx.info(cid)
        target = card_at(ctx.obs, opt.get("inPlayArea"), opt.get("inPlayIndex"), ctx.i)
        if target is None:
            return 400.0
        need = energy_need(target)
        if info is not None and info.type == TOOL:
            return 640.0 if not target["tools"] else -5.0
        is_active = opt.get("inPlayArea") == ACTIVE
        s = 780.0
        if need <= 0:
            s -= 250  # already powered
        elif need == 1:
            s += 80 if is_active else 50
        else:
            s += 40 if is_active else 25
        tinfo = ctx.info(target["id"])
        if info is not None and tinfo is not None and info.type == BASIC_ENERGY:
            # Prefer matching type for typed costs.
            if any(e == info.energy_type for a in tinfo.attacks for e in a.cost):
                s += 15
        return s
    if t == O_ABILITY:
        return 720.0
    if t == O_RETREAT:
        act = ctx.my_active
        if act is None or ctx.cur.get("retreated"):
            return -50.0
        bench_best = max((poke_readiness(ctx, b) for b in ctx.P["bench"]), default=-1e9)
        if bench_best > poke_readiness(ctx, act) + 60 and best_ready_damage(act) == 0:
            return 690.0
        if act["hp"] <= 0.25 * act["maxHp"] and bench_best > 0:
            return 680.0
        return -50.0
    if t == O_DISCARD:
        return -20.0
    return 100.0


def _hand_id(ctx, opt):
    c = card_at(ctx.obs, opt.get("area", HAND), opt.get("index"), ctx.i)
    return c["id"] if c else None


# ------------------------------------------------------- other selections

def score_card_option(ctx: Ctx, opt, context) -> float:
    t = opt["type"]
    player = opt.get("playerIndex", ctx.i)
    mine = player == ctx.i
    if t in (O_TOOL_CARD, O_ENERGY_CARD):
        poke = card_at(ctx.obs, opt.get("area"), opt.get("index"), player)
        surplus = -energy_need(poke) if poke else 0
        base = 5.0 * surplus
        if poke is not None and opt.get("area") == ACTIVE:
            base -= 5
        return base if mine else -base
    c = card_at(ctx.obs, opt.get("area"), opt.get("index"), player)
    cid = c["id"] if c else None

    if context in (C_SETUP_ACTIVE,):
        return poke_readiness(ctx, _as_poke(ctx, c))
    if context in (C_SWITCH, C_TO_ACTIVE):
        if mine:
            return 100 + poke_readiness(ctx, _as_poke(ctx, c))
        return target_value_for_damage(ctx, c, best_ready_damage(ctx.my_active) if ctx.my_active else 0)
    if context in (C_DMG_COUNTER, C_DMG_COUNTER_ANY, C_DAMAGE, C_EFFECT_TARGET, C_DETACH_FROM, C_DISABLE_ATTACK):
        if mine:
            return -50.0
        return target_value_for_damage(ctx, c)
    if context in (C_REMOVE_DMG, C_HEAL):
        if not mine or c is None:
            return -10.0
        return 1.0 + (c.get("maxHp", 0) - c.get("hp", 0))
    if context in (C_ATTACH_FROM, C_ATTACH_TO):
        # Which of our Pokémon gets the card: the one that most needs energy, active first.
        if c is None:
            return 1.0
        if "energies" in c:
            s = 10.0 * energy_need(c) + (5 if opt.get("area") == ACTIVE else 0)
            return s + 1 if mine else -s
        return card_value(ctx, cid)
    if context in LOSS_CONTEXTS:
        v = card_value(ctx, cid) if cid is not None else 20.0
        return -v if mine else v
    if context in GAIN_CONTEXTS or context in (C_EVOLVES_FROM,):
        if cid is None:
            return 5.0
        if context in (C_TO_BENCH, C_TO_FIELD, C_SETUP_BENCH):
            return 50.0 + 0.1 * (c.get("hp", 0) if isinstance(c, dict) else 0) if mine else -10.0
        return card_value(ctx, cid) if mine else 5.0
    # Unknown context: mild preference for our own cards, opponent's otherwise.
    return 1.0


def _as_poke(ctx, c):
    if c is None:
        return None
    if "energies" in c:
        return c
    info = ctx.info(c["id"])
    return {"id": c["id"], "hp": info.hp if info else 0, "maxHp": info.hp if info else 0, "energies": [], "tools": []}


def score_option(ctx: Ctx, idx: int) -> float:
    sel = ctx.select
    opt = sel["option"][idx]
    t = opt["type"]
    context = sel["context"]
    if context == C_MAIN:
        return score_main(ctx, opt)
    if t == O_NUMBER:
        n = opt.get("number", 0)
        return float(n) if context in (C_DRAW_COUNT, C_DMG_COUNTER_COUNT, C_REMOVE_DMG_COUNT) else -float(n)
    if t in (O_YES, O_NO):
        yes = t == O_YES
        if context == C_IS_FIRST:
            return 1.0 if yes else 0.0
        return 1.0 if yes else 0.0
    if t == O_ATTACK:
        a = ctx.db.attacks.get(opt.get("attackId"))
        if a is None:
            return 1.0
        return float(attack_value(a)) if context != C_DISABLE_ATTACK else float(attack_value(a))
    if t == O_SPECIAL:
        pr = CONDITION_PRIORITY.get(opt.get("specialConditionType"), 0)
        return float(pr)
    if t == O_ENERGY:
        # Paying a cost: keep energies that our Pokémon still needs; pay with surplus.
        poke = card_at(ctx.obs, opt.get("area"), opt.get("index"), opt.get("playerIndex", ctx.i))
        return -1.0 - (energy_need(poke) if poke else 0)
    if t == O_SKILL:
        return -float(idx)  # keep engine order
    if t in (O_CARD, O_TOOL_CARD, O_ENERGY_CARD):
        return score_card_option(ctx, opt, context)
    return 0.0


def score_all(obs) -> list[float]:
    ctx = Ctx(obs)
    return [score_option(ctx, i) for i in range(len(obs["select"]["option"]))]


def pick(select, scores, rng=None, temperature=0.0) -> list[int]:
    """Choose indices from scores respecting min/max counts.

    temperature > 0 samples a softmax-perturbed ranking (used for rollouts).
    """
    n = len(scores)
    lo, hi = select["minCount"], min(select["maxCount"], n)
    if temperature > 0 and rng is not None:
        keyed = [(s / temperature) - math.log(-math.log(rng.random() or 1e-12)) for s in scores]  # Gumbel
    else:
        keyed = scores
    order = sorted(range(n), key=lambda i: keyed[i], reverse=True)
    if select["context"] == C_MAIN or select["type"] in (6, 9, 8):  # single choice selects
        k = max(lo, min(hi, 1))
    else:
        k = sum(1 for s in scores if s > 0)
        k = max(lo, min(hi, k))
    return order[:k]


class HeuristicAgent(Agent):
    name = "heuristic"

    def act(self, obs):
        sel = obs["select"]
        if sel["context"] == C_MAIN and obs["current"]["turnActionCount"] > 60:
            # Safety valve against loops of repeatable actions.
            ends = [i for i, o in enumerate(sel["option"]) if o["type"] in (O_END, O_ATTACK)]
            if ends:
                return [ends[0]]
        return pick(sel, score_all(obs))

# ======================================================================
# ptcg/agents/search.py
# ======================================================================
"""Determinized Monte-Carlo search over the engine's forward model.

For each decision with more than one candidate action:
  1. candidates = every option for single-choice selects (MAIN, attack, yes/no,
     counts), or the heuristic pick plus sampled alternatives for multi-selects;
  2. repeat until the time/simulation budget runs out: sample a determinization
     of the hidden cards (ptcg.belief), apply a candidate (chosen by UCB1),
     roll out with the noisy heuristic policy for both players until
     `horizon` turn boundaries have passed (or the game ends), and score the
     final state with ptcg.evaluate;
  3. play the candidate with the best mean value.
Any engine/search error falls back to the heuristic pick, so the agent can
never return an illegal action because of the search.
"""


SINGLE_CHOICE_TYPES = {0, 6, 8, 9}  # MAIN, ATTACK, COUNT, YES_NO


class MCAgent(Agent):
    name = "mc"

    def __init__(self, deck, seed=None, budget=0.25, max_sims=400, horizon=1, temperature=40.0,
                 c_ucb=0.6, max_rollout_steps=400, alt_picks=4, min_remaining=60.0, weights=None,
                 search_non_main=True):
        super().__init__(deck, seed)
        self.budget = float(budget)
        self.max_sims = int(max_sims)
        self.horizon = int(horizon)
        self.temperature = float(temperature)
        self.c_ucb = float(c_ucb)
        self.max_rollout_steps = int(max_rollout_steps)
        self.alt_picks = int(alt_picks)
        self.min_remaining = float(min_remaining)
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.search_non_main = bool(search_non_main)
        self.belief = Belief(deck)
        self.fallback = HeuristicAgent(deck, seed)
        self.stats = {"searched": 0, "fallback": 0, "sims": 0, "errors": 0}

    def new_match(self):
        self.belief = Belief(self.deck)

    # ------------------------------------------------------------------
    def act(self, obs):
        self.belief.observe(obs)
        sel = obs["select"]
        heur = pick(sel, score_all(obs))
        if len(sel["option"]) <= 1 or sel["minCount"] == sel["maxCount"] == len(sel["option"]):
            return heur
        if sel["context"] != C_MAIN and not self.search_non_main:
            return heur
        if not obs.get("search_begin_input"):
            return heur
        remaining = float(obs.get("remainingOverageTime", 600.0) or 600.0)
        budget = self.budget if remaining > self.min_remaining else min(self.budget, 0.02)
        try:
            cands = self._candidates(obs, heur)
            if len(cands) <= 1:
                return heur
            best = self._search(obs, cands, budget)
            self.stats["searched"] += 1
            return best
        except Exception:
            self.stats["errors"] += 1
            return heur
        finally:
            try:
                end()
            except Exception:
                pass

    def _candidates(self, obs, heur):
        sel = obs["select"]
        n = len(sel["option"])
        if sel["type"] in SINGLE_CHOICE_TYPES and sel["maxCount"] == 1:
            return [[i] for i in range(n)]
        scores = score_all(obs)
        cands = [list(heur)]
        seen = {tuple(sorted(heur))}
        for _ in range(self.alt_picks * 3):
            if len(cands) > self.alt_picks:
                break
            c = pick(sel, scores, self.rng, temperature=self.temperature)
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                cands.append(c)
        return cands

    def _search(self, obs, cands, budget):
        me = obs["current"]["yourIndex"]
        root_turn = obs["current"]["turn"]
        n_opt = len(obs["select"]["option"])
        k = len(cands)
        visits = [0] * k
        total = [0.0] * k
        deadline = time.perf_counter() + budget
        sims = 0
        while sims < self.max_sims and (sims < k or time.perf_counter() < deadline):
            if sims < k:
                j = sims
            else:
                logn = math.log(sims)
                j = max(range(k), key=lambda a: total[a] / visits[a] + self.c_ucb * math.sqrt(logn / visits[a]))
            det = self.belief.determinize(obs, self.rng)
            sid, sobs = begin(obs, **det)
            if len(sobs["select"]["option"]) != n_opt:
                raise SearchError("determinized root differs from real root")
            v = self._rollout(sid, sobs, cands[j], me, root_turn)
            visits[j] += 1
            total[j] += v
            sims += 1
            if sims % 32 == 0:
                end()  # free memory of finished simulations
        self.stats["sims"] += sims
        best = max(range(k), key=lambda a: (total[a] / visits[a] if visits[a] else -9, visits[a]))
        return cands[best]

    def _rollout(self, sid, sobs, first_action, me, root_turn):
        sid, sobs = step(sid, first_action)
        steps = 0
        while True:
            cur = sobs["current"]
            if cur["result"] >= 0:
                break
            if cur["turn"] - root_turn >= self.horizon and sobs["select"]["context"] == C_MAIN:
                break
            if steps >= self.max_rollout_steps:
                break
            sel = sobs["select"]
            act = pick(sel, score_all(sobs), self.rng, temperature=self.temperature)
            sid, sobs = step(sid, act)
            steps += 1
        return evaluate(sobs, me, self.weights)



# =============================================================================== entry point
_DECK = [721, 722, 722, 722, 722, 723, 723, 723, 723, 1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1262, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1145, 1183, 1122, 1225, 1163]
_AGENT = MCAgent(_DECK, seed=None, budget=0.3, max_sims=400)


def agent(obs, config=None):
    """Kaggle entry point: returns the deck when obs["select"] is None, else option indices."""
    try:
        return _AGENT(obs)
    except Exception:
        # never lose the match to an exception: fall back to the heuristic, then to a legal default
        try:
            return HeuristicAgent(_DECK)(obs)
        except Exception:
            sel = obs.get("select")
            if sel is None:
                return list(_DECK)
            return list(range(sel["minCount"]))
