import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)


#enhancements blatantly stolen from Richard Lewis - well done that man!
# square feet per room feature -- take square footage and divide it by the number of bedrooms and bathrooms
df["sqft_per_room"] = df.apply(lambda row: (row.size_sqft // (row.bedrooms + row.bathrooms)) if row.bedrooms > 0 and row.bathrooms > 0 else row.size_sqft, axis=1)

# elevator to high floors feature -- you may not care about an elevator if your apartment is on the 2nd floor...you certainly do care if your apartment is on the 23rd
df["elev_to_high_floor"] = df.apply(lambda row: 1 if row.floor >= 4 and row.has_elevator == 1 else 0, axis=1)

# is luxury feature...if a building is newish and has a roof deck and a doorman, it's a luxury building ¯\_()_/¯ ---> teedles says: add a gym, roof deck perhaps not so much
df['is_luxury'] = df.apply(lambda row: 1 if row.building_age_yrs < 20 and row.has_gym == 1 and row.has_doorman == 1 else 0, axis=1)

# fire safety feature by Teedles: not ground floor and below 4th
# e.g. if something bad happens you can probably save yourself
df['is_firesafe'] = df.apply(lambda row: 1 if row.floor > 0 and row.floor < 4  else 0, axis=1)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'sqft_per_room', 'floor', 'building_age_yrs', 'has_doorman', 'has_elevator','elev_to_high_floor', 'has_dishwasher', 'has_gym', 'is_luxury', 'is_firesafe']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

lm = LinearRegression()

model = lm.fit(x_train, y_train)

y_predict= lm.predict(x_test)

print("Train score:")
print(lm.score(x_train, y_train))

print("Test score:")
print(lm.score(x_test, y_test))

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()