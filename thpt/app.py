from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

# Load data
df_data_map_region = pd.read_pickle("df_data_map_region.pkl")

app = Flask(__name__)

subjects = ['toan', 'ngu_van', 'ngoai_ngu', 'vat_li', 'hoa_hoc', 'sinh_hoc', 'lich_su', 'dia_li', 'gdcd']

# Predict average scores for the specified year
def predict_subject_scores(df_data_map_region, year):
    predictions = {'year': year}
    for subject in subjects:
        subject_data = df_data_map_region[['year', subject]].dropna()
        X = subject_data[['year']]
        y = subject_data[subject]
        model = LinearRegression()
        model.fit(X, y)
        predictions[subject] = model.predict([[year]])[0]
    return predictions

# Save predictions to CSV
predictions_file = 'predictions.csv'

def save_predictions(year, predictions):
    if os.path.exists(predictions_file):
        df = pd.read_csv(predictions_file, index_col=0)
        # Check if the prediction for the year already exists
        if year not in df['year'].values:
            new_row = pd.DataFrame([predictions])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(predictions_file)
    else:
        df = pd.DataFrame([predictions])
        df.to_csv(predictions_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_year = int(request.form.get('year'))
        predictions = predict_subject_scores(df_data_map_region, selected_year)
        save_predictions(selected_year, predictions)
        return redirect(url_for('index'))

    # Load predictions and combine with original data
    if os.path.exists(predictions_file):
        predictions_df = pd.read_csv(predictions_file, index_col=0)
        combined_data = pd.concat([df_data_map_region, predictions_df], ignore_index=True)
    else:
        combined_data = df_data_map_region

    # Ensure 'year' column is of integer type
    combined_data['year'] = combined_data['year'].astype(int)

    # Calculate average scores by year and differences
    avg_scores_by_year = combined_data.groupby('year')[subjects].mean().round(2).sort_index()
    latest_year = avg_scores_by_year.index[-1]
    prev_year = avg_scores_by_year.index[-2] if len(avg_scores_by_year.index) > 1 else None
    score_differences = (avg_scores_by_year.loc[latest_year] - avg_scores_by_year.loc[prev_year]).round(2).tolist() if prev_year else []

    # Plot predictions for each subject
    plt.figure(figsize=(10, 6))
    for subject in subjects:
        plt.plot(avg_scores_by_year.index, avg_scores_by_year[subject], label=subject, marker='o')
    plt.xlabel('Năm')
    plt.ylabel('Điểm trung bình')
    plt.title('Dự đoán điểm trung bình cho các môn học')
    plt.legend()
    plt.grid(True)

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()  # Close plot to avoid overlapping

    return render_template('index.html', avg_scores_by_year=avg_scores_by_year, 
                           score_differences=score_differences, latest_year=latest_year, prev_year=prev_year, 
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)