from os import environ

SESSION_CONFIGS = [
    dict(
        name='sportsbettingexp',
        display_name="Sports Betting and Household Finance",
        num_demo_participants=1,
        app_sequence=['sportsbettingexp'], 
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
    doc=""
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """
<h3>Welcome to the Sports Betting and Household Finance Experiment</h3>
<p>This is a project to understand bettor behavior mechanisms.</p>
"""

SECRET_KEY = environ.get('OTREE_SECRET_KEY', 'secret-key')
