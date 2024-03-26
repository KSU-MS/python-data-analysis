import sys
import requests
import logging
from parser_utils.download_latest_dbc_from_releases import repo_name,repo_owner

def check_newer_commit(dbc_repo_name, short_hash):
    url = f"https://api.github.com/repos/{dbc_repo_name}/compare/{short_hash}...HEAD"
    
    response = requests.get(url)
    logging.debug(f"repo url: {url}")
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "identical":
            logging.debug(f"No newer commit found after {short_hash}.")
            return False
        else:
            logging.warning(f"A newer DBC commit exists after {short_hash}. Your DBC may be out of date")
            return True
    else:
        logging.error("Failed to fetch repository data.")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.info("Usage: python script.py <repo_name> <short_hash>")
        sys.exit(1)

    repo_name = sys.argv[1]
    short_hash = sys.argv[2]

    check_newer_commit(repo_name, short_hash)
