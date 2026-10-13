import os, requests

cycles = {"2013": "H", "2015": "I", "2017": "J"}
files = ["DEMO", "BMX", "BPX", "BPQ", "DIQ", "SMQ", "PAQ", "MCQ"]
os.makedirs("data/raw/nhanes", exist_ok=True)

ok, failed = 0, []
for year, s in cycles.items():
    for f in files:
        url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{f}_{s}.xpt"
        path = f"data/raw/nhanes/{f}_{s}.xpt"
        if os.path.exists(path):
            ok += 1
            continue
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.ok:
            open(path, "wb").write(r.content)
            ok += 1
            print("saved", path)
        else:
            failed.append((url, r.status_code))
            print("FAILED", url, r.status_code)

print(f"\nDone: {ok}/24 downloaded")
for u, c in failed:
    print("Missing:", u, c)