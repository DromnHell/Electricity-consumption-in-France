LEGAL_PREFIX = ["L'ensemble des informations", "RTE ne pourra être tenu responsable"]

# RTE-France : Static URLs for ongoing consumption data
FRANCE_CONSUMPTION_CALENDAR_STATIC_URLS = [
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip",
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip"
]

# RTE-France : Calendar data URLs by year
FRANCE_CONSUMPTION_CALENDAR_BASE = "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_"
FRANCE_CONSUMPTION_CALENDAR_YEARS = ["2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021",
                                     "2022"]
FRANCE_CONSUMPTION_CALENDAR_URLS = [f"{FRANCE_CONSUMPTION_CALENDAR_BASE}{year}.zip" for year in
                                    FRANCE_CONSUMPTION_CALENDAR_YEARS]

# RTE-France : Tempo calendar URLs by season
FRANCE_TEMPO_CALENDAR_BASE = "https://eco2mix.rte-france.com/curves/downloadCalendrierTempo?season="
FRANCE_TEMPO_CALENDAR_YEARS = ["14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21", "21-22", "22-23",
                               "23-24", "24-25"]
FRANCE_TEMPO_CALENDAR_URLS = [f"{FRANCE_TEMPO_CALENDAR_BASE}{year}.zip" for year in FRANCE_TEMPO_CALENDAR_YEARS]

# Data-gouv : Daily departmental temperature (since January 2018)
FRANCE_DAILY_DEPARTMENTAL_TEMPERATURE_URL = "https://tabular-api.data.gouv.fr/api/resources/dd0df06a-85f2-4621-8b8b-5a3fe195bcd7/data/csv/"

# Data-gouv : Bank holidays (since 2005 to 2030)
FRANCE_BANK_HOLIDAYS_URL = "https://www.data.gouv.fr/fr/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5"

# School holidays (since 1990 to 2026)
FRANCE_SCHOOL_HOLIDAYS_URL = "https://raw.githubusercontent.com/AntoineAugusti/vacances-scolaires/master/data.csv"