import requests
import os
import sys
import logging

repo_owner = "KSU-MS"
repo_name = "ksu-ms-dbc"
download_dir = "dbc-files"
file_extension = ".dbc"  # Specify the file extension here

def download_latest_release(repo_owner=repo_owner, repo_name=repo_name, download_dir=download_dir, file_extension=file_extension):
    # Get the latest release information
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch latest release information: {response.status_code}")
        return

    release_info = response.json()
    assets = release_info.get('assets', [])
    if not assets:
        logging.warning("No assets found for the latest release")
        return

    # Download assets with the specified file extension
    for asset in assets:
        asset_url = asset.get('browser_download_url')
        if not asset_url:
            logging.error(f"No download URL found for asset: {asset.get('name')}")
            continue

        if asset_url.endswith(file_extension):
            download_path = os.path.join(download_dir, asset.get('name'))
            with open(download_path, 'wb') as f:
                asset_response = requests.get(asset_url)
                f.write(asset_response.content)
                logging.debug(f"asset['name] = {asset['name']}")
            logging.info(f"Downloaded asset '{asset['name']}' to: {download_path}")
                                            
if __name__ == "__main__":
    download_latest_release()
# Example usage

# print(sys.executable)
# download_latest_release(repo_owner, repo_name, download_dir, file_extension)
