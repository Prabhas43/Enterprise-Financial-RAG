import urllib.request
import os

os.makedirs("data", exist_ok=True)

# Download Apple's (AAPL) 10-K Financial Report from SEC EDGAR
url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'YourName StudentProject/1.0 (yourname@example.com)'}
)

with urllib.request.urlopen(req) as response, open("data/apple_10k.html", "wb") as out_file:
    out_file.write(response.read())

print(" SEC 10-K File downloaded successfully into /data folder!")