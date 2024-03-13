import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('rocket_vs_mflda.csv', index_col=0)
xs = df.index.to_numpy()

plt.ylim([0, 1])
plt.xlabel('Output Dimensionality')
plt.ylabel('F1-Score')
plt.title('Complete Data')

plt.plot(xs**2, df['MFLDA classifier'], 'o:', label='MUDRA')
plt.plot(xs**2, df['MFLDA with RidgeClassifierCV'], 's-.', label='MUDRA with RidgeClassifierCV')
plt.plot((xs**2 + 1) // 2 * 2, df['ROCKET'], '^--', label='ROCKET')
plt.legend()
plt.savefig('rocket_vs_mflda_completeData.svg')
# plt.savefig('rocket_vs_mflda_completeData.png')
plt.show()

plt.ylim([0, 1])
plt.xlabel('Output Dimensionality')
plt.ylabel('F1-Score')
plt.title('Missing Data')

plt.plot(xs**2, df['MFLDA classifier on missing data'], 'o:', label='MUDRA')
plt.plot(xs**2, df['MFLDA with RidgeClassifierCV on missing data'], 's-.', label='MUDRA with RidgeClassifierCV')
plt.plot((xs**2 + 1) // 2 * 2, df['ROCKET with imputation padding'], '^--', label='ROCKET with imputation padding')
plt.plot((xs**2 + 1) // 2 * 2, df['ROCKET with end padding'], 'P', linestyle=(0, (5, 2, 1, 2, 1, 2)) ,label='ROCKET with end padding')
plt.legend()
plt.savefig('rocket_vs_mflda_missingData.svg')
# plt.savefig('rocket_vs_mflda_missingData.png')
plt.show()
