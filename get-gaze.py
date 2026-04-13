#!/usr/bin/env python3
"""
Download gaze data for a list of users from LAreflecT API
and save EVERYTHING into one combined csv file:
    all_users_gaze.csv
"""

import os
import time
import json
import requests
from pathlib import Path

# ------------------ EDITABLE SETTINGS ---------------------

#SPACE_ID is the Activity ID
SPACE_ID   = "68f1e73de0971128e60ef539"
#SESSION_ID is the Task ID
SESSION_ID = "694233dae795683838f3b6c9"



USERS = [
    "68c23b1a7d52efd636d82154",
    "68c23af17d52efd636d81f51",
    "68c23af07d52efd636d81ed6",
    "68c23ae67d52efd636d81e49",
    "68c23af07d52efd636d81ee1",
    "68c23af17d52efd636d81f46",
    "68c23b2a7d52efd636d8222a",
    "68c23af37d52efd636d81fb7",
    "68c23aee7d52efd636d81e73",
    "68c23af17d52efd636d81f27",
    "68c23aef7d52efd636d81e9d",
    "68c23af07d52efd636d81ef1",
    "68c23af07d52efd636d81ef9",
    "68c23aef7d52efd636d81eb1",
    "68c23aef7d52efd636d81eb9",
    "68c23af07d52efd636d81ed2",
    "68c23aef7d52efd636d81ec9",
    "68c23af27d52efd636d81fa9",
    "68c23af07d52efd636d81ee9",
    "68c23aeb7d52efd636d81e65",
    "68c23aef7d52efd636d81e96",
    "68c23af07d52efd636d81f04",
    "68c23aef7d52efd636d81ec1",
    "68c23bdf7d52efd636d8267a",
    "68c23af27d52efd636d81f9b",
    "68c23b2b7d52efd636d82240",
    "68c23aee7d52efd636d81e7b",
    "68c23ae67d52efd636d81e51",
    "68c23afd7d52efd636d8204e",
    "68c23af57d52efd636d8200f",
    "68c23af77d52efd636d8202a",
    "68c23b4d7d52efd636d82322",
    "68c23af17d52efd636d81f32",
    "68c23af07d52efd636d81f0c",
    "68c23b217d52efd636d821b9",
    "68c23af07d52efd636d81f1c",
    "68c23af57d52efd636d82004",
    "68c23aef7d52efd636d81ea9",
    "68c23af07d52efd636d81f14",
    "68c23aee7d52efd636d81e8b",
    "68c23aee7d52efd636d81e83",
    "68c00ad69f56c49a83759bd9",
    # add additional userIds here...
]

BASE_URL = (
    "https://lareflectapi.edublab.net/"
    "eye-tracking/gaze/{space}/{session}"
)


PAGE_LIMIT = 1000
SORT = "-timestamp"
MAX_RETRIES = 5
BACKOFF = 2.0

OUTPUT_FILE = "all_users_gaze.csv"

# ----------------------------------------------------------

def headers():
    """Return HTTP headers with bearer token."""
    # token = os.getenv("LAREFLECT_TOKEN")
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0eXBlIjoic3lzdGVtIiwiY2xpZW50SWQiOiIwNGQ2NmY3OS0wOWEwLTQzYTMtOGVjZi04NWE3NGNiZDU4MjQiLCJpYXQiOjE3NzE5MTMyNjQsImV4cCI6MTc3MTkxNTA2NH0.qPkqmymSkJXsIBL4CTlGMl-jm2_FNsIj-xF-_uHw9uY"
    if not token:
        raise RuntimeError("Environment variable LAREFLECT_TOKEN is missing.")
    return {
        # "accept": "application/json",
        "accept": "text/csv",
        "x-system-authorization": f"Bearer {token}"
    }

def fetch_user(user_id: str):
    """Fetch all paginated results for one user."""
    url = BASE_URL.format(space=SPACE_ID, session=SESSION_ID)
    page = 1

    while True:
        params = {
            "userId": user_id,
            "limit": PAGE_LIMIT,
            "page": page,
            "sort": SORT
        }

        # retry loop
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, headers=headers(),
                                    params=params, timeout=30)

                if resp.status_code in (429,) or resp.status_code >= 500:
                    # backoff on server / rate limit
                    time.sleep(BACKOFF * (attempt + 1))
                    continue

                resp.raise_for_status()
                data = resp.json()
                break

            except Exception:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(BACKOFF * (attempt + 1))

        if not data:
            return

        # normalize formats
        if isinstance(data, dict) and "items" in data:
            items = data["items"]
        else:
            items = data

        if not isinstance(items, list):
            items = [items]

        # yield records one by one
        for entry in items:
            yield entry

        if len(items) < PAGE_LIMIT:
            return

        page += 1


def main():
    # prepare output file
    Path(OUTPUT_FILE).unlink(missing_ok=True)

    print(f"Writing combined output → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for uid in USERS:
            print(f"Fetching user: {uid}")
            for record in fetch_user(uid):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDone. All user records saved in:")
    print(f"  {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
