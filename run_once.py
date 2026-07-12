# run_once.py
# Place BOTH run_once.py and app_data.hex in C:\LoanFairnessProject
# Then run:  python run_once.py
import binascii

print("Reading hex data...")
with open("app_data.hex", "r") as f:
    hex_data = f.read().strip()

print("Writing app.py...")
data = binascii.unhexlify(hex_data)
with open("app.py", "wb") as f:
    f.write(data)

print(f"Done: {len(data):,} bytes written")
print("Now run:  streamlit run app.py")
