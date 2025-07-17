import base64

with open("fonts/riccione-serial-light-regular.ttf", "rb") as f:
    data = f.read()
    b64_encoded = base64.b64encode(data).decode("utf-8")
    print(f"data:font/ttf;base64,{b64_encoded}")
