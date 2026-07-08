import pandas as pd
import matplotlib.pyplot as plt

df_ridge=pd.read_csv("mse_ridge.csv")
df_lasso=pd.read_csv("mse_lasso.csv")

plt.figure()
plt.plot(df_ridge["lamda"], df_ridge["train_mse"], label="Ridge Train MSE")
plt.plot(df_ridge["lamda"], df_ridge["val_mse"], label="Ridge Val MSE")
plt.plot(df_lasso["lamda"], df_lasso["train_mse"], label="Lasso Train MSE", linestyle="--")
plt.plot(df_lasso["lamda"], df_lasso["val_mse"], label="Lasso Val MSE", linestyle="--")

plt.xscale("log")
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.legend()
plt.title("Training and Validation MSE vs Lambda (Ridge vs Lasso)")
plt.savefig("mse_vs_lambda_comparison.png")
plt.show()