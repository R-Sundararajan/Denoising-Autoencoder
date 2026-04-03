import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("evaluation_metrics.csv")

# Preview
print(df.head())

plt.figure()

plt.plot(df["AE_PSNR"].values, label="Autoencoder PSNR")
plt.plot(df["UNet_PSNR"].values, label="UNet PSNR")

plt.title("PSNR Comparison (AE vs UNet)")
plt.xlabel("Image Index")
plt.ylabel("PSNR (dB)")
plt.legend()

plt.show()

plt.figure()

plt.plot(df["AE_MSE"].values, label="Autoencoder MSE")
plt.plot(df["UNet_MSE"].values, label="UNet MSE")

plt.title("MSE Comparison (AE vs UNet)")
plt.xlabel("Image Index")
plt.ylabel("MSE")
plt.legend()

plt.show()

plt.figure()

plt.hist(df["AE_PSNR"], alpha=0.5, label="AE PSNR")
plt.hist(df["UNet_PSNR"], alpha=0.5, label="UNet PSNR")

plt.title("PSNR Distribution")
plt.xlabel("PSNR")
plt.ylabel("Frequency")
plt.legend()

plt.show()

plt.figure()

plt.scatter(df["AE_PSNR"], df["UNet_PSNR"])

plt.title("AE vs UNet PSNR (Per Image)")
plt.xlabel("AE PSNR")
plt.ylabel("UNet PSNR")

plt.show()
