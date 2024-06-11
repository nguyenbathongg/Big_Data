import pandas as pd
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

df = pd.read_csv('./dataset/water_potability.csv')

df.drop_duplicates(inplace=True)
df.dropna(how='all', inplace=True)

idx1 = df.query('Potability == 1')['ph'][df.ph.isna()].index
df.loc[idx1, 'ph'] = df.query('Potability == 1')['ph'][df.ph.notna()].mean()

idx0 = df.query('Potability == 0')['ph'][df.ph.isna()].index
df.loc[idx0, 'ph'] = df.query('Potability==0')['ph'][df.ph.notna()].mean()

idx1 = df.query('Potability == 1')['Sulfate'][df.Sulfate.isna()].index
df.loc[idx1, 'Sulfate'] = df.query('Potability == 1')['Sulfate'][df.Sulfate.notna()].mean()

idx0 = df.query('Potability == 0')['Sulfate'][df.Sulfate.isna()].index
df.loc[idx0, 'Sulfate'] = df.query('Potability==0')['Sulfate'][df.Sulfate.notna()].mean()

idx1 = df.query('Potability == 1')['Trihalomethanes'][df.Trihalomethanes.isna()].index
df.loc[idx1, 'Trihalomethanes'] = df.query('Potability == 1')['Trihalomethanes'][df.Trihalomethanes.notna()].mean()

idx0 = df.query('Potability == 0')['Trihalomethanes'][df.Trihalomethanes.isna()].index
df.loc[idx0, 'Trihalomethanes'] = df.query('Potability==0')['Trihalomethanes'][df.Trihalomethanes.notna()].mean()

df.loc[~df.ph.between(6.5, 8.5), 'Potability'] = 0

X = df.drop(['Potability'], axis=1).values
y = df['Potability'].values

# Chuẩn hóa dữ liệu
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)


# Huấn luyện mô hình Logistic Regression
# log_reg = LogisticRegression(random_state=42, max_iter=1000)
# log_reg.fit(X, y)


@app.route('/', methods=['GET', 'POST'])
def font():
    if request.method == 'POST':
        ph1 = request.form.get('_ph')
        ph = float(ph1)

        Hardness1 = request.form.get('_hardness')
        Hardness = float(Hardness1)

        Solids1 = request.form.get('_Solids')
        Solids = float(Solids1)

        Chloramines1 = request.form.get('_Chloramines')
        Chloramines = float(Chloramines1)

        Sulfate1 = request.form.get('_Sulfate')
        Sulfate = float(Sulfate1)

        Conductivity1 = request.form.get('_Conductivity')
        Conductivity = float(Conductivity1)  # Added for missing field

        Organic_carbon1 = request.form.get('_Organic_carbon')
        Organic_carbon = float(Organic_carbon1)

        Trihalomethanes1 = request.form.get('_Trihalomethanes')
        Trihalomethanes = float(Trihalomethanes1)

        Turbidity1 = request.form.get('_Turbidity')
        Turbidity = float(Turbidity1)
        # Potability = request.form.get('_Potability')  # Added for missing field


        # if (6 <= ph <= 9) and (120 <= Hardness <= 190) and (Solids <= 500) and (Chloramines <= 4) and (
        #         Sulfate <= 250) and (50 <= Conductivity <= 1500) and (Organic_carbon <= 2) and (
        #         Trihalomethanes <= 2) and (Turbidity <= 1):
        #     return render_template('good.html', message=message)
        # else:
        #     return render_template('bad.html', message=message)

        input_data = pd.DataFrame({'ph': [ph], 'Hardness': [Hardness], 'Solids': [Solids],
                                   'Chloramines': [Chloramines], 'Sulfate': [Sulfate], 'Conductivity': [Conductivity],
                                   'Organic_carbon': [Organic_carbon], 'Trihalomethanes': [Trihalomethanes],
                                   'Turbidity': [Turbidity]})

        # Scale the input data
        input_data = sc.transform(input_data)
        # Make a prediction
        prediction = rf.predict(input_data)
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

        # message = rf_accuracy

        if prediction == 0:
            return render_template('bad.html')
        else:
            return render_template('good.html')

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, port=6969)
