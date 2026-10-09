"""Generate a synthetic voice-assistant transcript dataset (template based).

Creates:
  data/annotated/intents.csv             text,intent   (gold labels)
  data/raw_transcripts/voice_queries.txt one raw transcript per line

The text imitates raw ASR output: no punctuation, random casing, spoken
fillers (um, uh, you know), informal contractions (gonna, wanna, lemme) and
occasional ASR word errors (dropped / duplicated words).

Run:  python scripts/make_dataset.py
"""
import csv
import random
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]

PLACES = ["the airport", "the train station", "city hospital", "the shopping mall",
          "my office", "home", "the central library", "the stadium"]
POIS = ["gas station", "charging station", "coffee shop", "parking lot", "pharmacy", "restaurant"]
GENRES = ["jazz", "rock", "classical", "pop", "country", "hip hop"]
PLAYLISTS = ["workout", "road trip", "chill", "party", "morning"]
PEOPLE = ["mom", "dad", "john", "sarah", "the office", "my brother", "my wife", "the doctor"]
TIMES = ["six thirty", "seven am", "eight fifteen", "noon", "five pm", "nine o clock"]
TASKS = ["buy milk", "pay the electricity bill", "pick up the kids", "water the plants",
         "renew my car insurance", "book a dentist appointment"]
ITEMS = ["eggs", "bread", "rice", "apples", "coffee", "batteries"]

T = {
    "Navigation": [
        "navigate to {p}", "take me to {p}", "give me directions to {p}", "how far is {p}",
        "find the nearest {poi}", "show traffic on the way to {p}",
        "avoid toll roads and take the fastest route", "start navigation to {p}",
        "cancel navigation", "what is my estimated arrival time", "find a {poi} near me",
        "reroute me around the traffic jam", "where is the closest {poi}"],
    "Media": [
        "play {g} music", "play the next song", "turn up the volume", "turn down the volume",
        "skip this track", "pause the podcast", "play my {pl} playlist",
        "shuffle my favorite songs", "stop the music", "play the news radio",
        "play the previous song", "mute the radio", "turn up the music", "resume the podcast"],
    "Climate": [
        "set the temperature to {n} degrees", "turn up the heat", "turn on the air conditioning",
        "turn off the heater", "make it warmer in here", "defrost the windshield",
        "turn on the seat heater", "set fan speed to {lv}", "turn off the air conditioning",
        "turn down the temperature", "make it cooler", "turn on the rear defroster",
        "turn the fan up"],
    "Phone": [
        "call {pe}", "dial {pe}", "send a text message to {pe}", "read my messages",
        "answer the call", "hang up the phone", "redial the last number",
        "call my credit card company", "call customer service", "reject the call",
        "text {pe} that i am running late", "read my voice mail", "call the credit card helpline"],
    "Reminder": [
        "set an alarm for {t}", "remind me to {task}", "set a timer for {n} minutes",
        "cancel my alarm", "add {it} to my shopping list", "what is on my calendar today",
        "schedule a meeting at {t}", "set a reminder to {task}", "wake me up at {t}",
        "turn off the alarm", "delete my reminder", "add a meeting to my calendar"],
}
PREFIX = ["", "", "", "hey", "okay", "please", "can you", "could you please",
          "i wanna", "lemme", "i need you to", "i'm gonna need you to"]
FILLERS = ["um", "uh", "you know", "i mean", "erm", "hmm"]


def slots():
    return dict(p=random.choice(PLACES), poi=random.choice(POIS), g=random.choice(GENRES),
                pl=random.choice(PLAYLISTS), pe=random.choice(PEOPLE), t=random.choice(TIMES),
                task=random.choice(TASKS), it=random.choice(ITEMS),
                n=random.choice(["20", "22", "24", "five", "ten", "15"]),
                lv=random.choice(["one", "two", "three", "high", "low"]))


def noisy(text):
    w = text.split()
    for _ in range(random.choice([0, 0, 1, 1, 2])):          # spoken fillers
        w.insert(random.randint(0, len(w)), random.choice(FILLERS))
    if random.random() < 0.08 and len(w) > 4:                # ASR deletion
        w.pop(random.randrange(len(w)))
    if random.random() < 0.05:                               # ASR duplication
        i = random.randrange(len(w))
        w.insert(i, w[i])
    s = " ".join(w)
    r = random.random()                                      # casing variation
    return s.upper() if r < 0.1 else s.capitalize() if r < 0.4 else s


def main():
    rows, seen = [], set()
    for intent, temps in T.items():
        n = 0
        while n < 70:
            base = random.choice(temps).format(**slots())
            text = noisy((random.choice(PREFIX) + " " + base).strip())
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            rows.append((text, intent))
            n += 1
    random.shuffle(rows)
    (ROOT / "data/annotated").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/raw_transcripts").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data/annotated/intents.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["text", "intent"])
        wr.writerows(rows)
    with open(ROOT / "data/raw_transcripts/voice_queries.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(r[0] for r in rows) + "\n")
    print(f"Wrote {len(rows)} transcripts")


if __name__ == "__main__":
    main()
