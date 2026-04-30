# ============================================================
# oTree Experiment: Mortgage Betting Game
# ============================================================
# This app simulates a financial decision-making experiment
# where participants manage savings, place bets over 25 rounds,
# and must cover a recurring "mortgage" payment each round.
# The goal is to study how people handle financial risk and
# debt accumulation under pressure.
# ============================================================

from otree.api import *  # Pull in everything oTree needs: Page, Player, models, etc.
import random            # We'll use this for resolving bet outcomes probabilistically


# ============================================================
# CONSTANTS — The fixed rules of the game
# ============================================================
# Think of this as the "rulebook" that never changes mid-game.
# oTree reads these values throughout the session.

class C(BaseConstants):
    NAME_IN_URL = 'sportsbettingexp'      # Shows up in the browser URL — keep it short and readable
    PLAYERS_PER_GROUP = None      # No grouping needed; each participant plays solo
    NUM_ROUNDS = 25               # The game runs for exactly 25 rounds (one "month" per round, conceptually)

    MORTGAGE_PAYMENT = cu(25)     # Every round, the player must pay cu(25) — this is the "mortgage"
                                  # If they can't cover it, it rolls into debt instead

    DEBT_LIMIT = cu(500)          # If a player accumulates cu(500) or more in debt, they go bankrupt
                                  # This is the hard ceiling


# ============================================================
# SUBSESSION & GROUP — Not used here, but required by oTree
# ============================================================
# oTree's architecture always expects these classes to exist.
# Since this is a solo game with no group interactions,
# we just pass through without adding anything custom.

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass


# ============================================================
# PLAYER — The data model for each participant
# ============================================================
# Every field here gets stored in oTree's database, one row per
# player per round. This is where we track everything that
# changes as the participant plays.

class Player(BasePlayer):

    # --- Betting inputs (collected from the participant) ---

    bet_amount = models.CurrencyField(
        label="Enter your bet amount:",
        min=cu(0),     # Can't bet negative
        max=cu(250),   # Hard cap at 250 so players can't go all-in beyond their starting balance
        initial=cu(0)  # Defaults to 0 if they don't enter anything
    )

    bet_odds = models.FloatField(
        label="Choose your bet odds",
        min=-400,  # Negative odds = favorite (e.g., -200 means bet 200 to win 100)
        max=400    # Positive odds = underdog (e.g., +400 means bet 100 to win 400)
                   # This mirrors the American odds system used in sports betting
    )

    # --- Derived / calculated fields (set by the app, not the player) ---

    bet_outcome = models.CurrencyField(initial=cu(0))  # Positive = won, Negative = lost
    savings     = models.CurrencyField(initial=cu(250)) # Everyone starts with cu(250) in round 1
    debt        = models.CurrencyField(initial=cu(0))   # Starts debt-free

    # --- Status flags ---

    mortgage_paid = models.BooleanField(initial=False)  # Did they cover this round's mortgage?
    bankrupt      = models.BooleanField(initial=False)  # Have they crossed the debt limit?


# ============================================================
# PROCESS BET — The core game logic
# ============================================================
# This function runs after the player submits their bet.
# It handles: carrying over balances, resolving the bet outcome,
# deducting the mortgage, and checking for bankruptcy.
# It's called in Bet.before_next_page() so it fires before
# the Results page is shown.

def process_bet(player: Player):

    # --- Step 1: Carry over savings and debt from the previous round ---
    # oTree creates a fresh Player record each round, so we need to
    # manually pull forward the balances. Round 1 uses the field defaults above.
    if player.round_number > 1:
        prev = player.in_round(player.round_number - 1)
        player.savings = prev.savings
        player.debt    = prev.debt

    # --- Step 2: Grab the bet inputs, defaulting safely if somehow missing ---
    bet  = player.bet_amount or cu(0)
    odds = player.bet_odds   or 0

    # --- Step 3: Convert American odds to a payout multiplier ---
    # American odds work differently depending on sign:
    #   Negative odds (favorite): you risk |odds| to win 100 → multiplier < 1
    #   Positive odds (underdog): you risk 100 to win odds → multiplier > 1
    if odds < 0:
        multiplier = 100 / abs(odds)  # e.g., -200 odds → multiplier = 0.5 (win half your bet)
    else:
        multiplier = odds / 100       # e.g., +300 odds → multiplier = 3.0 (win 3x your bet)

    # --- Step 4: Calculate win probability ---
    # We don't use true fair odds.
    # The formula tilts probability toward 50% but favors the house:
    #   - Negative odds (favorites) have slightly LOWER win probability than implied
    #   - Positive odds (underdogs) have slightly HIGHER win probability than implied
    # win_prob is clamped between 5% and 95% so outcomes never feel totally rigged.
    win_prob = max(0.05, min(0.95, 0.5 + (-odds / 800)))

    # --- Step 5: Resolve the bet ---
    if random.random() < win_prob:
        # Win: player gets their original bet back PLUS the winnings
        player.bet_outcome  = cu(bet * multiplier)       # Record the profit
        player.savings     += bet + player.bet_outcome   # Add back stake + profit
    else:
        # Loss: bet is simply deducted from savings
        player.bet_outcome  = -bet                       # Record the loss (as negative)
        player.savings     -= bet

    # --- Step 6: Collect the mortgage payment ---
    # Every round, cu(25) is due regardless of how the bet went.
    if player.savings >= C.MORTGAGE_PAYMENT:
        # Player can afford it — deduct and mark as paid
        player.savings       -= C.MORTGAGE_PAYMENT
        player.mortgage_paid  = True
    else:
        # Player can't cover it — whatever's short gets added to their debt
        unpaid         = C.MORTGAGE_PAYMENT - player.savings
        player.debt   += unpaid
        player.savings = cu(0)          # Savings bottoms out at zero, never goes negative
        player.mortgage_paid = False

    # --- Step 7: Check for bankruptcy ---
    # If debt has hit or exceeded the limit, flag the player as bankrupt.
    # Note: we track this flag but the current page_sequence doesn't gate on it —
    # you'd need to add an is_displayed() check if you want to redirect bankrupt players.
    player.bankrupt = player.debt >= C.DEBT_LIMIT


# ============================================================
# PAGES — What participants actually see
# ============================================================

# --- Introduction: Only shown in round 1 ---

class Introduction(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1  # Only appears at the very start


# --- Instructions: Also round 1 only ---

class Instructions(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1


# --- Bet: The main decision page, shown every round ---

class Bet(Page):
    form_model  = 'player'                        # Tells oTree which model to save form data to
    form_fields = ['bet_amount', 'bet_odds']      # The two inputs we collect from the player

    @staticmethod
    def vars_for_template(player: Player):
        # We pass custom variables to the HTML template so it can display
        # context-relevant information (current balance, allowed odds range, etc.)

        # Rounds 1–5 use a tighter odds range to ease players in.
        # After round 5, the full range opens up, introducing more risk/reward variance.
        if player.round_number <= 5:
            min_odds, max_odds = -200, 200   # Conservative range for early rounds
        else:
            min_odds, max_odds = -400, 400   # Full range for experienced players

        # Pull the current financial state from the previous round's data
        # (the current round's fields haven't been processed yet at this point)
        if player.round_number > 1:
            prev            = player.in_round(player.round_number - 1)
            current_savings = prev.savings
            current_debt    = prev.debt
            current_odds    = prev.bet_odds   # Show what they bet last round for reference
        else:
            # Round 1: use the initialized field defaults
            current_savings = player.savings
            current_debt    = player.debt
            current_odds    = 0               # No previous bet to reference

        return dict(
            ROUND_NUMBER  = player.round_number,
            NUM_ROUNDS    = C.NUM_ROUNDS,
            SAVINGS       = current_savings,
            DEBT          = current_debt,
            MIN_ODDS      = min_odds,
            MAX_ODDS      = max_odds,
            CURRENT_ODDS  = current_odds,     # Displayed as a reference point in the template
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # This hook fires after the player submits the form but BEFORE
        # they see the Results page. Perfect place to run our game logic.
        process_bet(player)


# --- Results: Feedback page shown after each bet is resolved ---
# Displays the outcome of this round and the player's updated financial state.

class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        # By the time this page loads, process_bet() has already run,
        # so all fields (savings, debt, bet_outcome, etc.) reflect this round's final state.
        return dict(
            ROUND_NUMBER  = player.round_number,
            NUM_ROUNDS    = C.NUM_ROUNDS,
            SAVINGS       = player.savings,
            DEBT          = player.debt,
            BET_OUTCOME   = player.bet_outcome,   # How much they won or lost this round
            BET_CHOICE    = player.bet_odds,       # The odds they chose
            BET_AMOUNT    = player.bet_amount,     # How much they wagered
            MORTGAGE_PAID = player.mortgage_paid,  # Did they cover the mortgage this round?
            BANKRUPT      = player.bankrupt,        # Are they over the debt limit?
            ALL_ROUNDS    = player.in_all_rounds()  # Full history — useful for showing a running chart
        )


# ============================================================
# PAGE SEQUENCE — The order participants move through the app
# ============================================================
# oTree loops through this sequence once per round.
# Introduction and Instructions self-filter to round 1 only via is_displayed().
# Bet and Results repeat every round.

page_sequence = [Introduction, Instructions, Bet, Results]
